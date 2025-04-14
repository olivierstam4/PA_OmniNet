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
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def preprocess(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.stack([image] * 3, axis=-1)
    image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0)
    return image

def preprocess_context(context_in, context_out):
    context_in_preprocessed = torch.stack([preprocess(c).squeeze(0) for c in context_in])
    context_out_preprocessed = torch.stack([preprocess(c).squeeze(0) for c in context_out])
    return context_in_preprocessed, context_out_preprocessed

class PreloadedH5Dataset(Dataset):
    def __init__(self, file_path, input_key, output_key, patient_ids):
        super().__init__()
        logging.info(f"Loading data from {file_path}...")
        with h5py.File(file_path, 'r') as data:
            full_data_input = np.array(data[input_key])
            full_data_output = np.array(data[output_key])
            full_patient_ids = np.array(data["patientID"])

        mask = np.isin(full_patient_ids, patient_ids)
        self.data_input = full_data_input[mask]
        self.data_output = full_data_output[mask]
        
        logging.info("Pre-processing all images...")
        self.processed_inputs = [preprocess(img) for img in tqdm(self.data_input, desc="Processing inputs")]
        self.processed_outputs = [preprocess(img) for img in tqdm(self.data_output, desc="Processing outputs")]
        logging.info("Data pre-processing complete")

    def get_context(self, context_size):
        """Return preprocessed context data of specified size"""
        if context_size > len(self.data_input):
            raise ValueError(f"Not enough data for context window size {context_size}")
        
        context_in = torch.stack([self.processed_inputs[i].squeeze(0) for i in range(context_size)])
        context_out = torch.stack([self.processed_outputs[i].squeeze(0) for i in range(context_size)])
        return context_in, context_out

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        return self.processed_inputs[idx].squeeze(), self.processed_outputs[idx].squeeze()

class LightningModel(pl.LightningModule):
    """
    We use pytorch lightning to organize our model code
    """

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

def test_model_with_context(context_size, model, dataset):
    context_in, context_out = dataset.get_context(context_size)
    
    ssim_vals, psnr_vals, rmse_vals = [], [], []
    
    csv_filename = f"context_csv_paper/PA_OmniNet_Generalized_MSFD_contextsize{context_size}.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "SSIM", "RMSE", "PSNR"])
        
        for idx in tqdm(range(len(dataset)), desc=f"Processing Context Size {context_size}"):
            target_in, y = dataset[idx]
            
            # Add batch dimension for model
            target_in = target_in.unsqueeze(0).to(DEVICE)
            y = y.unsqueeze(0).to(DEVICE)
            
            # Prepare context with batch dimension
            context_in_batch = context_in.unsqueeze(0).to(DEVICE)
            context_out_batch = context_out.unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                y_pred = model(target_in, context_in_batch, context_out_batch)
            
            torch.cuda.empty_cache()  # Free up GPU memory
            
            ssim_val = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)(y_pred, y).item()
            rmse_val = compute_rmse(y_pred, y)
            psnr_val = compute_psnr(y_pred, y)
            
            ssim_vals.append(ssim_val)
            rmse_vals.append(rmse_val)
            psnr_vals.append(psnr_val)
            writer.writerow([idx, ssim_val, rmse_val, psnr_val])
    
    return np.mean(ssim_vals), np.mean(rmse_vals), np.mean(psnr_vals)

if __name__ == "__main__":
    hparams = {
        "nb_levels": 4,
        "nb_inner_channels": 32,
        "nb_conv_layers_per_stage": 2,
        "data_slice_only": True,
    }
    
    file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/MSFD_multisegment_ss_RawBP.h5"
    input_key = 'ms,ss32_BP_w780'
    output_key = 'ms_BP_w780' 
    
    #model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/Alpha/PA_OmniNet_MSFD.ckpt"
    model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/Alpha/PA_OmniNet_Generalized.ckpt"

    model = LightningModel.load_from_checkpoint(checkpoint_path=model_path)
    model.to(DEVICE)
    model.eval()
    

    with h5py.File(file_path, 'r') as h5_file:
        patient_ids = sorted(set(map(int, h5_file['patientID'][:])))
    test_patient_ids = patient_ids[-2:]
    
    test_dataset = PreloadedH5Dataset(file_path, input_key, output_key, test_patient_ids)
    
    ssim_results, rmse_results, psnr_results = [], [], []
    
    for context_size in tqdm(range(1, 33), desc='Testing context sizes'):
        tqdm.write(f"Testing with context size: {context_size}")
        ssim_avg, rmse_avg, psnr_avg = test_model_with_context(context_size, model, test_dataset)
        ssim_results.append(ssim_avg)
        rmse_results.append(rmse_avg)
        psnr_results.append(psnr_avg)
        logging.info(f"Context size {context_size}: SSIM={ssim_avg:.4f}, RMSE={rmse_avg:.4f}, PSNR={psnr_avg:.4f}")
    

    logging.info(f"SSIM scores: {ssim_results}")
    logging.info(f"RMSE scores: {rmse_results}")
    logging.info(f"PSNR scores: {psnr_results}")