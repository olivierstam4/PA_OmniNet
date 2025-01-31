# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
import os
from unet1 import UNet
import logging
from tqdm import tqdm
import numpy as np
logging.basicConfig(
    filename="training_log.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger()
logger.addHandler(console_handler)

HDF5_INPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_semicircle_RawBP.h5"
HDF5_OUTPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_semicircle_RawBP.h5"
INPUT_KEY = "sc,ss32_BP"
OUTPUT_KEY = "sc_BP"

GAMMA = 0.98
MOMENTUM = 0.9
STEP_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 250
LOSS_MUL = 1e4
num_patches = 100
n_samples = 1000
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
BATCH_SIZE = 128
START_FILTERS = 32
MODEL_SAVE_PATH = "unet_model_SWFD_patient.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cuda_available = torch.cuda.is_available()
current_device = torch.cuda.current_device() if cuda_available else None
gpu_name = torch.cuda.get_device_name(current_device) if cuda_available else "No GPU"
cpu_count = os.cpu_count()

logger.info(f"Available CPU cores: {cpu_count}")
logger.info(f"CUDA Available: {cuda_available}")
logger.info(f"Current CUDA Device: {current_device if cuda_available else 'N/A'}")
logger.info(f"GPU Name: {gpu_name}")
num_workers = 18


class OADATDataloader(Dataset):
    def __init__(self, input_file_path, output_file_path, input_key, output_key, patient_ids=None, normalize=True):
        logger.info(f"Loading dataset for patients {patient_ids} from {input_file_path} and {output_file_path}...")
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.input_key = input_key
        self.output_key = output_key
        self.patient_ids = patient_ids
        self.normalize = normalize
        with h5py.File(input_file_path, 'r') as f_input, h5py.File(output_file_path, 'r') as f_output:
            self.all_patient_ids = f_input["patientID"][:]
            if patient_ids is None or len(patient_ids) == 0:
                mask = np.ones(len(self.all_patient_ids), dtype=bool)
            else:
                mask = np.isin(self.all_patient_ids, patient_ids)


            self.indices = np.where(mask)[0]
            logger.info(f"Loaded {len(self.indices)} samples for patients {patient_ids}.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.input_file_path, 'r') as f_input, h5py.File(self.output_file_path, 'r') as f_output:
            real_idx = self.indices[idx]
            x = torch.tensor(f_input[self.input_key][real_idx], dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(f_output[self.output_key][real_idx], dtype=torch.float32).unsqueeze(0)

            if self.normalize:
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                y = (y - y.min()) / (y.max() - y.min() + 1e-8)

        return x, y


if __name__ == '__main__':
    logger.info("Starting the data loading process...")
    EARLY_STOPPING_PATIENCE = 50
    best_val_loss = float('inf')
    early_stop_counter = 0
    with h5py.File(HDF5_INPUT_PATH, 'r') as f:
        all_patient_ids = f["patientID"][:]
    sorted_patient_ids = np.sort(np.unique(all_patient_ids))

    test_patient_ids = sorted_patient_ids[-2:]
    val_patient_ids = sorted_patient_ids[-4:-2]
    train_patient_ids = sorted_patient_ids[:-4]
    logger.info("Different patients: {sorted_patient_ids} \nTraining on patients {train_patient_ids}\nValidating on patients {val_patient_ids}\nTesting on patients {test_patient_ids} ")
    train_dataset = OADATDataloader(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, INPUT_KEY,
                                    OUTPUT_KEY, patient_ids=train_patient_ids)
    val_dataset = OADATDataloader(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, INPUT_KEY,
                                  OUTPUT_KEY, patient_ids=val_patient_ids)
    test_dataset = OADATDataloader(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, INPUT_KEY,
                                   OUTPUT_KEY, patient_ids=test_patient_ids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=num_workers,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, persistent_workers=True)

    logger.info(
        f"Data loading complete. Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples, Test: {len(test_dataset)} samples.")

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{EPOCHS}...")

        model.train()
        train_loss = 0
        logger.info(f"Starting training for Epoch {epoch + 1}...")
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Training]") as pbar:
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = F.mse_loss(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)
        train_loss /= len(train_loader)
        logger.info(f"Training complete for Epoch {epoch + 1}. Average Training Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0
        logger.info(f"Starting validation for Epoch {epoch + 1}...")
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch + 1}/{EPOCHS} [Validation]") as pbar:
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    y_pred = model(x)
                    loss = F.mse_loss(y_pred, y)
                    val_loss += loss.item()
                    pbar.set_postfix({"Batch Loss": loss.item()})
                    pbar.update(1)
        val_loss /= len(val_loader)
        logger.info(f"Validation complete for Epoch {epoch + 1}. Average Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model_Semi.pth")
            logger.info(f"Validation loss improved. Model saved as best_model_Semi.pth.")
        else:
            early_stop_counter += 1
            logger.info(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")
    
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered. Stopping training.")
            break

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join("checkpoints", f"Unet_SWFDsemi_patient_{epoch + 1}.ckpt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    model.load_state_dict(torch.load("best_model.pth"))
    logger.info("Loaded the best model for testing.")

    logger.info("Starting testing phase...")
    model.eval()
    test_loss = 0
    with tqdm(total=len(test_loader), desc="Testing") as pbar:
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_pred = model(x)
                loss = F.mse_loss(y_pred, y)
                test_loss += loss.item()
                pbar.update(1)
    test_loss /= len(test_loader)
    logger.info(f"Testing complete. Average Test Loss: {test_loss:.4f}")
