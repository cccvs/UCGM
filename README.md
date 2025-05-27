
<h1 align="center"> UCGM: Unified Continuous Generative Models </h1>

<p align="center">
  Peng&nbsp;Sun<sup>1,2</sup> &ensp; <b>&middot;</b> &ensp;
  Yi&nbsp;Jiang<sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://tlin-taolin.github.io/" target="_blank">Tao&nbsp;Lin</a><sup>1</sup> &ensp; &ensp;
</p>

<p align="center">
  <sup>1</sup>Westlake University &emsp; <sup>2</sup>Zhejiang University&emsp; <br>
</p>

<p align="center">
<a href="https://huggingface.co/sp12138sp/UCGM">:robot: Models</a> &ensp;
<a href="https://arxiv.org/abs/2505.07447">:page_facing_up: Paper</a> &ensp;
<a href="#label-bibliography">:label: BibTeX</a> &ensp;
  <br><br>
<a href="https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=unified-continuous-generative-models"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unified-continuous-generative-models/image-generation-on-imagenet-256x256" alt="PWC"></a> <a href="https://paperswithcode.com/sota/image-generation-on-imagenet-512x512?p=unified-continuous-generative-models"><img src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unified-continuous-generative-models/image-generation-on-imagenet-512x512" alt="PWC"></a>
</p>

Official PyTorch implementation of **UCGM** **T**rainer and **S**ampler (**UCGM-{T,S}**): :trophy: A unified framework for training, sampling, and understanding continuous generative models (including diffusion, flow-matching, consistency models).

<div align="center">
  <img src="assets/fig1_a_512.png" width="48%">
  <img src="assets/fig1_b_512.png" width="48%">
  <p>
    <strong>Generated samples from two 675M diffusion transformers trained with UCGM on ImageNet-1K 512×512.</strong><br>
    Left: A multi-step model (Steps=NFE=40, FID=1.48) | Right: A few-step model (Steps=NFE=2, FID=1.75)<br>
    <em>Samples generated without classifier-free guidance or other guidance techniques.</em>
  </p>
</div>


## :sparkles: Features

:rocket: **Plug-and-Play Acceleration**: UCGM-S *boosts various pre-trained multi-step continuous models for free*—e.g., given a model from [REPA-E](https://github.com/End2End-Diffusion/REPA-E) (on ImageNet 256×256):  
- :white_check_mark: **Cuts 84% of sampling steps (Steps=250 → Steps=40) while improving FID (1.26 → 1.06)**  
  :white_check_mark: **Training-free** and **no additional cost introduced**  

- :bar_chart: **Extended results** for more accelerated models are available [here](#fast_forward-ucgm-s-plug-and-play-acceleration).  

:zap: **Lightning-Fast Model Tuning**: UCGM-T transforms any pre-trained multi-step continuous model (e.g., REPA-E with FID=1.54 at NFE=80) into a **high-performance, few-step generator** with **record efficiency**:  
- :white_check_mark: **FID=1.39 @ Steps=NFE=2 (ImageNet-1K 256×256)**  
  :white_check_mark: **Tuned in just 8 minutes on 8 GPUs**  

- :bar_chart: **Extended results** for additional tuned models are available [here](#zap-ucgm-t-ultra-efficient-tuning-system).  

:fire: **Efficient Unified Framework**: Train/sample diffusion, flow-matching, and consistency models in one system, outperforming peers at low steps:  
- :white_check_mark: **FID=1.21 @ Steps=NFE=30 (ImageNet 256×256), 1.48 FID @ Steps=NFE=40 on 512×512**  
  :white_check_mark: Just **2 steps**? Still strong (**1.42 FID on 256×256, 1.75 FID on 512×512**)  
  :white_check_mark: No classifier-free guidance or other techniques—**simpler and faster**  
  :white_check_mark: Compatible with diverse datasets (ImageNet, CIFAR, etc.) and architectures (CNNs, Transformers)—**high flexibility**  

- :bar_chart: **Extended results** for additional trained models are available [here](#gear-ucgm-ts-efficient-unified-framework).  

:book: Check more detailed features in our [paper](https://arxiv.org/abs/2505.07447)!  


## :gear: Preparation

1. Download necessary files from [Huggingface](https://huggingface.co/sp12138sp/UCGM/tree/main), including:
   - Checkpoints of various VAEs
   - Statistic files for datasets
   - Reference files for FID calculation

2. Place the downloaded `outputs` and `buffers` folders at the same directory level as this `README.md`

3. For dataset preparation (skip if not training models), run:
```bash
bash scripts/data/in1k256.sh
```

## :rocket: UCGM-S: Plug-and-Play Acceleration

Accelerate any continuous generative model (diffusion, flow-matching, etc.) with UCGM-S. Results marked with :rocket: denote UCGM-S acceleration.  
*NFE = Number of Function Evaluations (sampling computation cost)*

| Method                                                  | Model Size | Dataset  | Resolution      | NFE           | FID  | NFE (🚀)    | FID (🚀) | Model                                                   |
| ------------------------------------------------------- | ---------- | -------- | --------------- | ------------- | ---- | ------------- | ------- | ------------------------------------------------------- |
| [REPA-E](https://github.com/End2End-Diffusion/REPA-E)   | 675M       | ImageNet | 256×256         | 250×2         | 1.26 | 40×2          | 1.06    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [Lightning-DiT](https://github.com/hustvl/LightningDiT) | 675M       | ImageNet | 256×256         | 250×2         | 1.35 | 50×2          | 1.21    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DDT](https://github.com/MCG-NJU/DDT)                   | 675M       | ImageNet | 256×256         | 250×2         | 1.26 | 50×2          | 1.27    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [EDM2-S](https://github.com/NVlabs/edm2)                | 280M       | ImageNet | 512×512         | 63            | 2.56 | 40            | 2.53    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [EDM2-L](https://github.com/NVlabs/edm2)                | 778M       | ImageNet | 512×512         | 63            | 2.06 | 50            | 2.04    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [EDM2-XXL](https://github.com/NVlabs/edm2)              | 1.5B       | ImageNet | 512×512         | 63            | 1.91 | 40            | 1.88    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DDT](https://github.com/MCG-NJU/DDT)                   | 675M       | ImageNet | 512×512         | 250×2         | 1.28 | 150×2         | 1.18    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |

**:computer: Usage Examples**: Generate images and evaluate FID using a REPA-E trained model:
```bash
# Generate samples using public pretrained multi-step model
bash scripts/run_eval.sh ./configs/sampling_multi_steps/in1k256_sit_xl_repae_linear.yaml
```

## :zap: UCGM-T: Ultra-Efficient Tuning System

UCGM-T revolutionizes multi-step generative models (including diffusion and flow matching models) by enabling ultra-efficient conversion to high-performance few-step versions. Results marked with :zap: indicate UCGM-T-tuned models.

| Pre-trained Model                                       | Model Size | Dataset  | Resolution | Tuning Efficiency  | NFE (⚡) | FID (⚡) | Tuned Model                                             |
| ------------------------------------------------------- | ---------- | -------- | ---------- | -------------------- | ------- | ------- | ------------------------------------------------------- |
| [Lightning-DiT](https://github.com/hustvl/LightningDiT) | 675M       | ImageNet | 256×256    | 0.64 epoch (10 mins) | 2       | 2.06    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [REPA](https://github.com/sihyun-yu/REPA)               | 675M       | ImageNet | 256×256    | 0.64 epoch (13 mins) | 2       | 1.95    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [REPA-E](https://github.com/End2End-Diffusion/REPA-E)   | 675M       | ImageNet | 256×256    | 0.40 epoch (8 mins)  | 2       | 1.39    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DDT](https://github.com/MCG-NJU/DDT)                   | 675M       | ImageNet | 256×256    | 0.32 epoch (11 mins) | 2       | 1.90    | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |

(Please note that the tuning time mentioned above is based on evaluation using 8 H800 GPUs)

**:computer: Usage Examples**

Generate Images:
```bash
# Generate samples using our tuned few-step model
bash scripts/run_eval.sh ./configs/tuning_few_steps/in1k256_sit_xl_repae_linear.yaml
```

Tune Models:
```bash
# Tune a multi-step model into few-step version
bash scripts/run_train.sh ./configs/tuning_few_steps/in1k256_sit_xl_repae_linear.yaml
```


## :fire: UCGM-{T,S}: Efficient Unified Framework

Train multi-step and few-step models (diffusion, flow-matching, consistency) with UCGM-T. All models sample efficiently using UCGM-S without guidance.

| Encoders                                                                            | Model Size | Resolution  | Dataset  | NFE | FID  | Model                                                   |
| ----------------------------------------------------------------------------------- | ---------- | ----------- | -------- | --- | ---- | ------------------------------------------------------- |
| [VA-VAE](https://github.com/hustvl/LightningDiT)                                    | 675M       | 256×256     | ImageNet | 30  | 1.21 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [VA-VAE](https://github.com/hustvl/LightningDiT)                                    | 675M       | 256×256     | ImageNet | 2   | 1.42 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DC-AE](https://github.com/mit-han-lab/efficientvit/tree/master/applications/dc_ae) | 675M       | 512×512     | ImageNet | 40  | 1.48 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |
| [DC-AE](https://github.com/mit-han-lab/efficientvit/tree/master/applications/dc_ae) | 675M       | 512×512     | ImageNet | 2   | 1.75 | [Link](https://huggingface.co/sp12138sp/UCGM/tree/main) |

**:computer: Usage Examples**

Generate Images:
```bash
# Generate samples using our pretrained few-step model
bash scripts/run_eval.sh ./configs/training_few_steps/in1k256_tit_xl_vavae.yaml
```

Train Models:
```bash
# Train a new multi-step model (full training)
bash scripts/run_train.sh ./configs/training_multi_steps/in1k512_tit_xl_dcae.yaml

# Convert to few-step model (requires pretrained multi-step checkpoint)
bash scripts/run_train.sh ./configs/training_few_steps/in1k512_tit_xl_dcae.yaml
```

:exclamation: **Note for few-step training**:
1. Requires initialization from a multi-step checkpoint
2. Prepare your checkpoint file with both `model` and `ema` keys:
   ```python
   {
     "model": multi_step_ckpt["ema"], 
     "ema": multi_step_ckpt["ema"]
   }
   ```


## :label: Bibliography

If you find this repository helpful for your project, please consider citing our work:

```
@article{sun2025unified,
  title = {Unified continuous generative models},
  author = {Sun, Peng and Jiang, Yi and Lin, Tao},
  journal = {arXiv preprint arXiv:2505.07447},
  year = {2025},
  url = {https://arxiv.org/abs/2505.07447},
  archiveprefix = {arXiv},
  eprint = {2505.07447},
  primaryclass = {cs.LG}
}
```


## :page_facing_up: License

Apache License 2.0 - See [LICENSE](LICENSE) for details.