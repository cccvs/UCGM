import warnings

warnings.filterwarnings("ignore")

import torch
import torch.backends.cuda
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import random
import argparse
import json
import time
import itertools
from time import time
from copy import deepcopy
from accelerate import Accelerator
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, UNet2DConditionModel
from PIL import Image

# local imports
from networks import NETWORKS
from optimers import OPTIMERS
from autoencs import AUTOENCS
from methodes import METHODES
from utilities import ImgLatentDataset
from utilities import create_logger, load_config
from utilities import set_seed, update_ema, remove_module_prefix, remove_module_all

from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        text = item["text"]
        image_file = item["image_file"]
        
        # read image
        raw_image = Image.open(os.path.join(self.image_root_path, image_file))
        image = self.transform(raw_image.convert("RGB"))
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values
        
        # drop
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1
        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        
        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)
    

def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]

    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

def get_model(train_config, accelerator):
    pretrained_model_name_or_path = train_config['model']["pretrained_model_name_or_path"]
    image_encoder_path = train_config['model']["image_encoder_path"]
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # ip-adapter
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )
    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, train_config['model']["pretrained_ip_adapter_path"])
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)

    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(),  ip_adapter.adapter_modules.parameters())

    return ip_adapter, params_to_opt, tokenizer, vae, text_encoder, image_encoder



def do_train(train_config, accelerator):
    """
    Training a generative model.
    """
    # Setup accelerator:
    device = accelerator.device
    rank = accelerator.process_index
    seed = train_config["train"]["global_seed"] * accelerator.num_processes + rank
    set_seed(seed)

    experiment_dir = f"{train_config['output_dir']}/{train_config['exp_name']}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(train_config["output_dir"], exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, "train")
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    downsample_ratio = train_config["vae"]["downsample_ratio"]
    assert (
        train_config["data"]["image_size"] % downsample_ratio == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = train_config["data"]["image_size"] // downsample_ratio
    # Load scheduler, tokenizer and models.

    # model = NETWORKS[train_config["model"]["type"]](
    #     input_size=latent_size,
    #     num_classes=train_config["data"]["num_classes"],
    #     in_channels=(train_config["model"]["in_chans"]),
    # ).to(device)

    model, params_to_opt, tokenizer, vae, text_encoder, image_encoder = get_model(train_config, accelerator)
    model = model.to(device)
    params_to_opt_list = list(params_to_opt)
    print(f"[{rank}] {len(params_to_opt_list)}, params to optimize")
    
    ema = deepcopy(model).requires_grad_(False).to(device)

    if accelerator.is_main_process:
        demoimages_dir = f"{experiment_dir}/demoimages"
        os.makedirs(demoimages_dir, exist_ok=True)

        if train_config["data"]["image_size"] == 256:
            demo_y = torch.tensor([975, 3, 207, 387, 388, 88, 979, 279], device=device)
        elif train_config["data"]["image_size"] == 512:
            demo_y = torch.tensor([975, 207], device=device)
        elif train_config["data"]["image_size"] == 32:
            demo_y = torch.tensor(list(range(0, 16)), device=device)
        # demo_z = torch.randn(
        #     len(demo_y), model.in_channels, latent_size, latent_size, device=device
        # )
        demo_z = torch.randn(
            len(demo_y), image_encoder.config.projection_dim, latent_size, latent_size, device=device
        )

        # vae = AUTOENCS[train_config["vae"]["type"]](train_config["vae"]["type"])
        
        logger.info("Loaded VAE model")

    unigen = METHODES["unigen"](
        transport_type=train_config["transport"]["type"],
        lab_drop_ratio=train_config["transport"]["lab_drop_ratio"],
        consistc_ratio=train_config["transport"]["consistc_ratio"],
        enhanced_ratio=train_config["transport"]["enhanced_ratio"],
        enhanced_style=train_config["transport"]["enhanced_style"],
        scaled_cbl_eps=train_config["transport"]["scaled_cbl_eps"],
        ema_decay_rate=train_config["transport"]["ema_decay_rate"],
        enhanced_range=train_config["transport"]["enhanced_range"],
        time_dist_ctrl=train_config["transport"]["time_dist_ctrl"],
        wt_cosine_loss=train_config["transport"]["wt_cosine_loss"],
        weight_funcion=train_config["transport"]["weight_funcion"],
    )
    if accelerator.is_main_process:
        logger.info(
            f"SD15 parameters: {sum(p.numel() for p in params_to_opt_list) / 1e6:.2f}M"
        )
        logger.info(
            f"Optimizer: {train_config['optimizer']['type']}, lr={train_config['optimizer']['lr']}, beta1={train_config['optimizer']['beta1']}, beta2={train_config['optimizer']['beta2']}"
        )
        logger.info(f'Use cosine loss: {train_config["transport"]["wt_cosine_loss"]}')
        logger.info(f'Use weight func: {train_config["transport"]["weight_funcion"]}')

    opt = OPTIMERS[train_config["optimizer"]["type"]](
        params_to_opt_list,
        lr=train_config["optimizer"]["lr"],
        weight_decay=train_config["optimizer"]["weight_decay"],
        betas=(train_config["optimizer"]["beta1"], train_config["optimizer"]["beta2"]),
    )

    # tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    dataset = MyDataset(
        json_file=train_config["data"]["json_file"],
        tokenizer=tokenizer,
        size=train_config["data"]["image_size"],
        image_root_path=train_config["data"]["image_root_path"]
    )
    
    batch_size_per_gpu = (
        train_config["train"]["global_batch_size"] // accelerator.num_processes
    )

    global_batch_size = batch_size_per_gpu * accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        shuffle=True,
        num_workers=train_config["data"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )


    if accelerator.is_main_process:
        mean, stad, latent_multiplier = (
            dataset._latent_mean.cuda(),
            dataset._latent_std.cuda(),
            dataset.latent_multiplier,
        )
        logger.info(
            f"Dataset contains {len(dataset):,} images {train_config['data']['data_path']}"
        )
        logger.info(
            f"Batch size {batch_size_per_gpu} per gpu, with {global_batch_size} global batch size"
        )

    if "ckpt" in train_config["train"]:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
        model.load_state_dict(checkpoint["model"])
        if train_config["train"]["no_reopt"] is not True:
            opt.load_state_dict(checkpoint["opt"])
        if train_config["train"]["no_reuni"] is not True:
            if ((unigen.cor > 0.0) or (unigen.enr > 0.0)) and unigen.emd > 0.0:
                unigen.mod = deepcopy(model).requires_grad_(False).to(device)
            unigen.load_state_dict(checkpoint["unigen"])
        ema.load_state_dict(checkpoint["ema"])
        train_steps = int(checkpoint_path.split("/")[-1].split(".")[0])
        del checkpoint
        if accelerator.is_main_process:
            logger.info(f"Loaded checkpoint at: {checkpoint_path}.")
    else:
        train_steps = 0
        update_ema(ema, model, decay=0)
        if accelerator.is_main_process:
            logger.info("Starting training from scratch.")

    # Prepare models for training:
    model.train()
    ema.eval()
    if train_config["train"]["no_buffer"] is True:
        model = DDP(model, device_ids=[rank], broadcast_buffers=False)
    model, opt, loader, unigen = accelerator.prepare(model, opt, loader, unigen)

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    while True:
        for x, y in loader:
            if accelerator.mixed_precision == "no":
                x = x.to(device, dtype=torch.float32)
                y = y.to(device)
            else:
                x = x.to(device)
                y = y.to(device)

            loss = unigen.training_step(model, x, y)

            opt.zero_grad()
            accelerator.backward(loss)
            if "max_grad_norm" in train_config["optimizer"]:
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), train_config["optimizer"]["max_grad_norm"]
                    )
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
            update_ema(ema, model, train_config["train"]["ema_decay"])

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % train_config["train"]["log_every"] == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint:
            if (
                train_steps % train_config["train"]["ckpt_every"] == 0
                and train_steps > 0
            ):
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": remove_module_prefix(model.state_dict()),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "unigen": remove_module_all(unigen.state_dict()),
                        "config": train_config,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    demox = unigen.sampling_loop(demo_z, ema, **dict(y=demo_y))
                    demox = demox[1::2].reshape(-1, *demox.shape[2:])
                    demox = (demox * stad) / latent_multiplier + mean
                    demox = vae.decode_to_images(demox).cpu()
                    demoimages_path = f"{demoimages_dir}/{train_steps:07d}.png"
                    save_image(demox, os.path.join(demoimages_path), nrow=len(demo_y))
                    logger.info(f"Saved demoimages to {demoimages_path}")
                    del checkpoint, demox

                dist.barrier()

            if train_steps >= train_config["train"]["max_steps"]:
                break
        if train_steps >= train_config["train"]["max_steps"]:
            break
    if accelerator.is_main_process:
        logger.info("Done!")

    return accelerator


if __name__ == "__main__":
    # read config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/debug.yaml")
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)
    par_path = args.config.split("/")
    train_config["exp_name"] = os.path.join(par_path[-2], par_path[-1].split(".")[0])
    do_train(train_config, accelerator)
