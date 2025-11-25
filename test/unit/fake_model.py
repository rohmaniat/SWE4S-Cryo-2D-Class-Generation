'''Starting by creating a fake model and fake dataset
Which is needed for unit tests'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class FakeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.tensor(1.0))

    def forward(self, images, targets=None):
        # Loss must depend on dummy_param so autograd can trace backward
        loss1 = self.dummy_param * 1.0
        loss2 = self.dummy_param * 0.5

        return {
            "loss_classifier": loss1,
            "loss_box_reg": loss2,
        }


class FakeCryoDataset(Dataset):
    def __len__(self):
        return 4   # small for testing

    def __getitem__(self, idx):
        # Fake image: 1×8×8 tensor
        image = torch.zeros((1, 8, 8), dtype=torch.float32)

        # Fake coordinates: two particle centers
        coords = torch.tensor([[2, 2], [5, 5]], dtype=torch.float32)

        return image, coords


def fake_collate(batch):
    images = [item[0] for item in batch]
    coords = [item[1] for item in batch]

    images = torch.stack(images)
    return images, coords
