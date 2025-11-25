import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import unittest
import test.unit.fake_model

sys.path.append('src/')  # noqa

import utils


class TestTrainOneEpoch(unittest.TestCase):

    def test_train_one_epoch_runs(self):

        model = test.unit.fake_model.FakeDetectionModel()

        # Load in my fake dataset
        dataset = test.unit.fake_model.FakeCryoDataset()
        data_loader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=test.unit.fake_model.fake_collate)

        # Simple optimizer to simulate weighting
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        device = torch.device("cpu")

        avg_loss = utils.train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            criterion=None,
            device=device,
            epoch_num=0
        )

        # Expected: fixed losses → (1.0 + 0.5) = 1.5 per batch
        # 4 samples, batch_size=2 → 2 batches → avg should be ~1.5

        self.assertGreater(avg_loss, 0)
        self.assertLess(avg_loss, 2.0)
        self.assertAlmostEqual(avg_loss, 1.5, delta=0.2)
