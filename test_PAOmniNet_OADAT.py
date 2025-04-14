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


class CustomH5Dataset(Dataset):
    def __init__(self, file_path, context_size=16, input_key=None,
                 output_key=None, patient_ids=None, context_patient_id=None):
        super().__init__()
        self.file_path = file_path
        self.context_size = context_size
        self.input_key = input_key
        self.output_key = output_key

        logging.info(f"Loading data from {file_path}...")
        with h5py.File(file_path, 'r') as data:
            full_data_input = np.array(data[input_key])
            full_data_output = np.array(data[output_key])
            full_patient_ids = np.array(data["patientID"])

        if patient_ids is not None:
            mask = np.isin(full_patient_ids, patient_ids)
            self.data_input = full_data_input[mask]
            self.data_output = full_data_output[mask]
            self.patient_ids = full_patient_ids[mask]
        else:
            self.data_input = full_data_input
            self.data_output = full_data_output
            self.patient_ids = full_patient_ids

        if context_patient_id is not None:
            context_mask = full_patient_ids == context_patient_id
            if not np.any(context_mask):
                raise ValueError(f"No data found for context patient ID {context_patient_id}.")
            if np.sum(context_mask) < context_size:
                raise ValueError(f"Not enough data for context from patient ID {context_patient_id}.")
            self.global_context_in = full_data_input[context_mask][:context_size]
            self.global_context_out = full_data_output[context_mask][:context_size]
            self.context_patient_ids = full_patient_ids[context_mask][:context_size]
            logging.info(f"Context taken from patient ID {context_patient_id}.")
        else:
            self.global_context_in = self.data_input[:context_size]
            self.global_context_out = self.data_output[:context_size]
            self.context_patient_ids = self.patient_ids[:context_size]
            logging.warning("No context patient ID provided. Using global context.")

        assert len(self.data_input) >= context_size, "Insufficient data for context window."
        logging.info(
            f"Loaded {len(self.data_input)} samples for patients {patient_ids} from {file_path}.")

    def __len__(self):
        return len(self.data_input)

    def __getitem__(self, idx):
        context_in = self.global_context_in
        context_out = self.global_context_out

        #logging.info(f"Context patient IDs: {self.context_patient_ids}")

        X = preprocess(self.data_input[idx])
        y = preprocess(self.data_output[idx])
        context_in, context_out = preprocess_context(context_in, context_out)

        return X.squeeze(), y.squeeze(), context_in, context_out



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

    def on_test_start(self):
        """Initialize the CSV file before testing starts."""
        self.csv_file = "PAOmniNet_Generalized_alpha_Multi_id1.csv"
        #self.csv_file = "MSFD_baseline.csv"
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Index", "SSIM", "RMSE", "PSNR"])  

    def test_step(self, batch, batch_idx):
        """Compute metrics and append them to the CSV file."""
        target_in, y, context_in, context_out = batch
        i = 0
        y_pred = self(target_in, context_in, context_out)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
        ssim_val = ssim(y_pred, y).item()
        rmse_val = compute_rmse(y_pred, y)
        psnr_val = compute_psnr(y_pred, y)
        #logging.info(i, ssim_val)
        #i += 1
        #ssim_val = ssim(target_in, y).item()
        #rmse_val = compute_rmse(target_in, y)
        #psnr_val = compute_psnr(target_in, y)
        with open(self.csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([batch_idx, ssim_val, rmse_val, psnr_val])

        self.log("test_ssim", ssim_val, prog_bar=True)
        self.log("test_rmse", rmse_val, prog_bar=True)
        self.log("test_psnr", psnr_val, prog_bar=True)
    def on_test_end(self):
        """Compute and print the average SSIM, RMSE, and PSNR after testing."""
        import pandas as pd
        
        df = pd.read_csv(self.csv_file)
        
        avg_ssim = df["SSIM"].mean()
        avg_rmse = df["RMSE"].mean()
        avg_psnr = df["PSNR"].mean()
        
        logging.info(f"Average SSIM: {avg_ssim:.4f}")
        logging.info(f"Average RMSE: {avg_rmse:.4f}")
        logging.info(f"Average PSNR: {avg_psnr:.4f}")

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
    
    # file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_semicircle_RawBP.h5"
    # input_key = "sc,ss32_BP"
    # output_key = "sc_BP"
    
    #file_path = "/gpfs/home3/ostam/Datasets/oadat/swfd/SWFD_semicircle_RawBP-mini.h5"
     
    # file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/MSFD_multisegment_ss_RawBP.h5"
    # input_key ='ms,ss32_BP_w780'
    # output_key = 'ms_BP_w780' 
    
    file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_multisegment_ss_RawBP.h5"
    input_key ='ms,ss32_BP'
    output_key = "ms_BP"

    batch_size = 1
    
    model = LightningModel(hparams)  
    # Generalized
    model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/Alpha/PA_OmniNet_Generalized.ckpt"
    # model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/MSE/PA_OmniNet_Generalized.ckpt"
    # model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/SSIM/PA_OmniNet_Generalized.ckpt"

    # Multi
    #model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/Alpha/PA_OmniNet_Multi.ckpt"
    # model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/MSE/PA_OmniNet_Multi.ckpt"
    # model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/SSIM/PA_OmniNet_Multi.ckpt"

    # MSFD
    # model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/Alpha/PA_OmniNet_MSFD.ckpt"
    # model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/MSE/PA_OmniNet_MSFD.ckpt"
    # model_path = "/gpfs/home3/ostam/weights/PAOmni/OADAT/SSIM/PA_OmniNet_MSFD.ckpt"

    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/final_model_semifinalpatient_early.pth"
    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/final_model_multifinalpatient_early.pth"
    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpointsrevison/PAOmni_ssimmse_Generalized_mseval-epoch=205-val_loss=0.0010.ckpt"
    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpointsrevison/PAOmni_ssimmse_Generalized-epoch=249-val_loss=0.1084.ckpt"
    # model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpointsrevison/PAOmni_ssim_Generalized-epoch=242-val_loss=0.1068.ckpt"
    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/final_model_MFSDfinalpatient_early.pth"
     # Load the state dictionary from the .pth file
    #state_dict = torch.load(model_path, map_location=torch.device(
        #'cuda' if torch.cuda.is_available() else 'cpu'))
    #model.load_state_dict(state_dict)
    model = LightningModel.load_from_checkpoint(checkpoint_path=model_path)
    # Set device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)

    # model = LightningModel.load_from_checkpoint(model_path)
    # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    from torchmetrics.image import StructuralSimilarityIndexMeasure

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    # model.eval()

    with h5py.File(file_path, 'r') as h5_file:
        patient_ids = h5_file['patientID'][:] 
        total_patient_ids = sorted(
            set(map(int, patient_ids)))  

    test_patient_ids = total_patient_ids[-2:] 
    val_patient_ids = total_patient_ids[
                      -4:-2]  
    train_patient_ids = total_patient_ids[:-4]  
    logging.info(
        f"Different patients: {total_patient_ids} \nTraining on patients {train_patient_ids}\nValidating on patients {val_patient_ids}\nTesting on patients       {test_patient_ids} ")

    logging.info(f"Training IDs: {train_patient_ids}")
    logging.info(f"Validation IDs: {val_patient_ids}")
    logging.info(f"Testing IDs: {test_patient_ids}")
    logging.info(f"Testing {model_path}\nData: {file_path}\nIn/Output: {input_key}, {output_key}")
    test_dataset = CustomH5Dataset(
        file_path=file_path,
        input_key=input_key,
        output_key=output_key,
        context_size=16,
        patient_ids=test_patient_ids,
        context_patient_id=1
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=16, persistent_workers=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False 
    )
    test_results = trainer.test(model, test_loader)
    logging.info(f"Test results: {test_results}")
