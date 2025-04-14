from testing_functions import *
import pytorch_lightning as pl
from models.pairwise_conv_avg_model import PairwiseConvAvgModel
from util.shapecheck import ShapeChecker
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import logging
import csv
from torchmetrics.image import StructuralSimilarityIndexMeasure

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def preprocess(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.stack([image] * 3, axis=-1)
    image = torch.tensor(image).float()
    image = image.permute(2, 0, 1).unsqueeze(0)
    return image


def preprocess_context(context_in, context_out):
    context_in_preprocessed = np.array([preprocess(c) for c in context_in])
    context_out_preprocessed = np.array([preprocess(c) for c in context_out])

    context_in_preprocessed = torch.tensor(context_in_preprocessed).squeeze(1)
    context_out_preprocessed = torch.tensor(context_out_preprocessed).squeeze(1)

    return context_in_preprocessed, context_out_preprocessed


class OADATDataLoader(Dataset):
    def __init__(self, file_path, context_size=4, input_key=None,
                 output_key=None, patient_ids=None):
        super().__init__()
        self.file_path = file_path
        self.context_size = context_size
        self.input_key = input_key
        self.output_key = output_key

        logging.info(f"Loading data from {file_path}...")
        with h5py.File(file_path, 'r') as data:
            self.data_input = np.array(data[input_key])
            self.data_output = np.array(data[output_key])
            self.patient_ids = np.array(data["patientID"])

        if patient_ids is not None:
            mask = np.isin(self.patient_ids, patient_ids)
            self.data_input = self.data_input[mask]
            self.data_output = self.data_output[mask]
            self.patient_ids = self.patient_ids[mask]

        assert len(
            self.data_input) >= context_size + 1, "Insufficient data for context and target."
        logging.info(f"Loaded {len(self.data_input)} samples for patients {patient_ids} from {file_path}.")

    def __len__(self):
        return len(self.data_input) - self.context_size

    def __getitem__(self, idx):
        X = preprocess(self.data_input[idx])
        y = preprocess(self.data_output[idx])
        context_in = self.data_input[idx + 1: idx + 1 + self.context_size]
        context_out = self.data_output[idx + 1: idx + 1 + self.context_size]
        context_in, context_out = preprocess_context(context_in, context_out)

        return X.squeeze(), y.squeeze(), context_in, context_out


class LightningModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.net = PairwiseConvAvgModel(
            dim=2 if self.hparams.data_slice_only else 3,
            stages=self.hparams.nb_levels,
            in_channels=3,
            out_channels=3,
            inner_channels=self.hparams.nb_inner_channels,
            conv_layers_per_stage=self.hparams.nb_conv_layers_per_stage)

    def forward(self, target_in, context_in, context_out):
        sc = ShapeChecker()
        target_in = target_in.squeeze(1)
        sc.check(target_in, "B C H W", C=3, H=256, W=256)
        sc.check(context_in, "B L C H W")
        sc.check(context_out, "B L C H W")

        y_pred = self.net(context_in, context_out, target_in)
        sc.check(y_pred, "B C H W")

        return y_pred