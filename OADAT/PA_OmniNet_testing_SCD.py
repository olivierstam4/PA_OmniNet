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


class OADATDataloader_SCD(Dataset):
    def __init__(self, file_path_in, file_path_out, input_key, output_key, indices, context_size=16):
        super().__init__()
        self.file_path_in = file_path_in
        self.file_path_out = file_path_out
        self.input_key = input_key
        self.output_key = output_key
        self.indices = indices
        self.context_size = context_size

        logging.info(f"Loading data from {file_path_in} and {file_path_out}...")

        with h5py.File(file_path_in, 'r') as f_in, h5py.File(file_path_out, 'r') as f_out:
            full_data_input = np.array(f_in[input_key])
            full_data_output = np.array(f_out[output_key])
            
        self.global_context_in = full_data_input[(context_size * 1):(context_size * 2)]
        self.global_context_out = full_data_output[(context_size * 1):(context_size * 2)]
        self.data_input = full_data_input[indices]
        self.data_output = full_data_output[indices]

        assert len(self.data_input) >= context_size + 1, "Insufficient data for context and target."
        logging.info(f"Loaded {len(self.data_input)} samples from the provided indices.")

        self.context_in = self.global_context_in
        self.context_out = self.global_context_out
        self.context_in, self.context_out = preprocess_context(self.context_in, self.context_out)

    def __len__(self):
        return len(self.data_input) - self.context_size

    def __getitem__(self, idx):
        X = preprocess(self.data_input[idx + self.context_size])
        y = preprocess(self.data_output[idx + self.context_size])

        return X.squeeze(), y.squeeze(), self.context_in, self.context_out


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

    def on_test_start(self):
        self.csv_file = "PAOmniNet_Specific_SCD_secondcontext.csv"
        #self.csv_file = "SCD_baseline.csv"
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "SSIM", "RMSE", "PSNR"])

    def test_step(self, batch, batch_idx):
        target_in, y, context_in, context_out = batch
        y_pred = self(target_in, context_in, context_out)
        logging.info(f"{y.shape}, {y_pred.shape}")

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
        #ssim_val = ssim(y_pred, y).item()
        #rmse_val = compute_rmse(y_pred, y)
        #psnr_val = compute_psnr(y_pred, y)
        ssim_val = ssim(target_in, y).item()
        rmse_val = compute_rmse(target_in, y)
        psnr_val = compute_psnr(target_in, y)

        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([batch_idx, ssim_val, rmse_val, psnr_val])

        self.log("test_ssim", ssim_val, prog_bar=True)
        self.log("test_rmse", rmse_val, prog_bar=True)
        self.log("test_psnr", psnr_val, prog_bar=True)

if __name__ == "__main__":
    hparams = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "nb_levels": 4,
        "nb_inner_channels": 32,
        "nb_conv_layers_per_stage": 2,
        "data_slice_only": True,
        "max_epochs": 250,
    }

    file_path_in = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SCD_multisegment_ss_RawBP.h5"
    file_path_out = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SCD_RawBP.h5"
    input_key = 'ms,ss32_BP'
    output_key = "ms_BP"

    with h5py.File(file_path_in, 'r') as f:
        total_samples = len(f[input_key])
    
    train_split = int(0.7 * total_samples)
    val_split = int(0.2 * total_samples)
    train_indices = np.arange(0, train_split)
    val_indices = np.arange(train_split, train_split + val_split)
    test_indices = np.arange(train_split + val_split, total_samples)

    logging.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    BATCH_SIZE = 1

    test_dataset = OADATDataloader_SCD(file_path_in, file_path_out, input_key, output_key, test_indices, context_size=16)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)

    model = LightningModel(hparams)
    model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpoints/final_model_SCDfinalpatient_early.pth"
    #model_path ="/gpfs/home3/ostam/export_snellius/neuralizer/final_model_semifinalpatient_early.pth"

    state_dict = torch.load(model_path, map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False
    )

    test_results = trainer.test(model, test_loader)
    logging.info(f"Test results: {test_results}")