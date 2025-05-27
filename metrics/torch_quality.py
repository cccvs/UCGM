import torch
import torch_fidelity
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FIDDataset(Dataset):
    def __init__(self, data):

        self.images = data
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x * 255),
                transforms.Lambda(lambda x: x.to(torch.uint8)),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img


def torch_quality_evaluate(data, fid_reference_file, output_dir=None):
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=FIDDataset(data),
        input2=None,
        fid_statistics_file=fid_reference_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=True,
    )
    return metrics_dict
