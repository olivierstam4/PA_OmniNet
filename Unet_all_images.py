import h5py
import numpy as np
from skimage.metrics import structural_similarity as ssim
import math
import torch
from torch.utils.data import DataLoader, Dataset
from unet1 import UNet
from testing_functions import compute_rmse, compute_psnr
import csv
import logging
from torchmetrics.image import StructuralSimilarityIndexMeasure

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#HDF5_INPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_semicircle_RawBP.h5"
#HDF5_OUTPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_semicircle_RawBP.h5"
#INPUT_KEY = 'sc,ss32_BP'
#OUTPUT_KEY = 'sc_BP'
#HDF5_INPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_multisegment_ss_RawBP.h5"
#HDF5_OUTPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_multisegment_ss_RawBP.h5"
#INPUT_KEY ='ms,ss32_BP'
#OUTPUT_KEY = "ms_BP"
HDF5_INPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SCD_multisegment_ss_RawBP.h5"
HDF5_OUTPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SCD_RawBP.h5"
INPUT_KEY ='ms,ss32_BP'
OUTPUT_KEY = "ms_BP"
#MODEL_PATH = "/gpfs/home3/ostam/Unet/checkpoints/Unet_SWFDsemi_patient_250.ckpt"
#MODEL_PATH = "/gpfs/home3/ostam/Unet/checkpoints/Unet_SWFDsemi_patient_150.ckpt"
#MODEL_PATH = "/gpfs/home3/ostam/Unet/checkpoints/Unet_MFSD2_patient_250.ckpt"
#MODEL_PATH ="/gpfs/home3/ostam/Unet/checkpoints/Unet_MFSD2_patient_150.ckpt"
#MODEL_PATH ="/gpfs/home3/ostam/Unet/checkpoints/Unet_MFSD2_patient_50.ckpt"
#MODEL_PATH = "/gpfs/home3/ostam/Unet/checkpoints/Unet_multi2_patient_200.ckpt"
#HDF5_INPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/MSFD_multisegment_ss_RawBP.h5"
#HDF5_OUTPUT_PATH = "/gpfs/scratch1/shared/tmp.zlixKwrks6/MSFD_multisegment_ss_RawBP.h5"
#INPUT_KEY ='ms,ss32_BP_w780'
#OUTPUT_KEY = "ms_BP_w780"
# input_key = "sc,ss32_BP"
# output_key = "sc_BP"
#HDF5_INPUT_PATH = '../Datasets/OADAT/swfd/SWFD_semicircle_RawBP-mini.h5'
#HDF5_OUTPUT_PATH = '../Datasets/OADAT/swfd/SWFD_semicircle_RawBP-mini.h5'
# INPUT_KEY ='ms,ss32_BP_w780'
# OUTPUT_KEY = "ms_BP_w780"
file_path = '../Datasets/OADAT/swfd/SWFD_semicircle_RawBP-mini.h5'
CSV_OUTPUT_PATH = "metricsSemi_on_MFSDearly.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ssim = StructuralSimilarityIndexMeasure().to(DEVICE)

# class DataloaderMouse(Dataset):
#     def __init__(self, input_file_path, output_file_path, input_key, output_key, patient_ids=None, normalize=True):
#         self.input_file_path = input_file_path
#         self.output_file_path = output_file_path
#         self.input_key = input_key
#         self.output_key = output_key
#         self.patient_ids = patient_ids
#         self.normalize = normalize
# 
#         with h5py.File(input_file_path, 'r') as f_input, h5py.File(output_file_path, 'r') as f_output:
#             self.all_patient_ids = f_input["patientID"][:]
#             if patient_ids is None or len(patient_ids) == 0:
#                 mask = np.ones(len(self.all_patient_ids), dtype=bool)  # Include all
#             else:
#                 mask = np.isin(self.all_patient_ids, patient_ids)  # Filter by patient IDs
#             self.indices = np.where(mask)[0]
# 
#     def __len__(self):
#         return len(self.indices)
# 
#     def __getitem__(self, idx):
#         with h5py.File(self.input_file_path, 'r') as f_input, h5py.File(self.output_file_path, 'r') as f_output:
#             real_idx = self.indices[idx]
#             x = torch.tensor(f_input[self.input_key][real_idx], dtype=torch.float32).unsqueeze(0)
#             y = torch.tensor(f_output[self.output_key][real_idx], dtype=torch.float32).unsqueeze(0)
# 
#             if self.normalize:
#                 x = (x - x.min()) / (x.max() - x.min() + 1e-8)
#                 y = (y - y.min()) / (y.max() - y.min() + 1e-8)
# 
#         return x, y



class OADATDataloader(Dataset):
    def __init__(self, input_file_path, output_file_path, input_key, output_key, indices, normalize=True):
        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.input_key = input_key
        self.output_key = output_key
        self.indices = indices
        self.normalize = normalize

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        with h5py.File(self.input_file_path, 'r') as f_input, h5py.File(self.output_file_path, 'r') as f_output:
            x = torch.tensor(f_input[self.input_key][real_idx], dtype=torch.float32).unsqueeze(0)
            y = torch.tensor(f_output[self.output_key][real_idx], dtype=torch.float32).unsqueeze(0)

            if self.normalize:
                x = (x - x.min()) / (x.max() - x.min() + 1e-8)
                y = (y - y.min()) / (y.max() - y.min() + 1e-8)

        return x, y


#model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVIC

MODEL_PATH = "/gpfs/home3/ostam/Unet/best_model_Semi.pth"
#MODEL_PATH = "final_model_weight/Unet/best_model_Semi.pth"
#MODEL_PATH = "/gpfs/home3/ostam/Unet/best_model_MFSD.pth"# Update to use the new model checkpoint
#MODEL_PATH = "/gpfs/home3/ostam/Unet/best_model_Multi.pth"
#MODEL_PATH ="/gpfs/home3/ostam/Unet/best_model_SCD.pth"
logging.info(f"Using device: {DEVICE} on model {MODEL_PATH}")
logging.info("Loading the model...")
model = UNet().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)))
    logging.info(f"Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise
model.eval()

#with h5py.File(HDF5_INPUT_PATH, 'r') as f:
 #   all_patient_ids = f["patientID"][:]  # Assuming patient IDs are stored as a column

#sorted_patient_ids = np.sort(np.unique(all_patient_ids))

#test_patient_ids = sorted_patient_ids[-2:]  # Last two for testing
#val_patient_ids = sorted_patient_ids[-4:-2]  # Second-to-last two for validation
#train_patient_ids = sorted_patient_ids[:-4]
#logging.info(f"Different patients: {sorted_patient_ids} \nTraining on patients {train_patient_ids}\nValidating on patients {val_patient_ids}\nTesting on patients       {test_patient_ids} ")

#test_dataset = DataloaderMouse(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, INPUT_KEY,
#                           OUTPUT_KEY, patient_ids=test_patient_ids)

#test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
#                            num_workers=6, persistent_workers=True)

with h5py.File(HDF5_INPUT_PATH, 'r') as f:
         total_samples = len(f[INPUT_KEY])
train_split = int(0.7 * total_samples)
val_split = int(0.2 * total_samples)
train_indices = np.arange(0, train_split)
val_indices = np.arange(train_split, train_split + val_split)
test_indices = np.arange(train_split + val_split, total_samples)
logging.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
BATCH_SIZE = 1
test_dataset = OADATDataloader(HDF5_INPUT_PATH, HDF5_OUTPUT_PATH, INPUT_KEY, OUTPUT_KEY, test_indices)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6)
import os
import matplotlib.pyplot as plt


def save_grayscale_image(image, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()


def test_step(batch, batch_idx, output_dir):
    input_img, ground_truth_img = batch
    with torch.no_grad():
        y_pred = model(input_img.to(DEVICE))
    input_img = input_img.squeeze().cpu().numpy()
    ground_truth_img = ground_truth_img.squeeze().cpu().numpy()
    y_pred = y_pred.squeeze().cpu().numpy()
    if input_img.ndim == 3:
        input_img = input_img.mean(axis=0)
    if ground_truth_img.ndim == 3:
        ground_truth_img = ground_truth_img.mean(axis=0)
    if y_pred.ndim == 3:
        y_pred = y_pred.mean(axis=0)
    output_dir = os.path.join(output_dir, "U-net Generalized")
    save_grayscale_image(y_pred, output_dir, f"unet_generalized_{batch_idx}.png")

    #output_dir = os.path.join(output_dir, "U-net Specific")
    #save_grayscale_image(y_pred, output_dir, f"unet_specific_{batch_idx}.png")

if __name__ == "__main__":
    output_dir = "./SCD"
    for batch_idx, (input_img, ground_truth_img) in enumerate(test_loader):
        test_step((input_img, ground_truth_img), batch_idx, output_dir)




