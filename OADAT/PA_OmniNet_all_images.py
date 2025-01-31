import h5py
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


class OADATDataloader(Dataset):
    def __init__(self, file_path, context_size=4, input_key=None,
                 output_key=None, patient_ids=None):
        super().__init__()
        self.file_path = file_path
        self.context_size = context_size
        self.input_key = input_key
        self.output_key = output_key

        logging.info(f"Loading data from {file_path}...")
        with h5py.File(file_path, 'r') as data:
            full_data_input = np.array(data[input_key])
            full_data_output = np.array(data[output_key])
            self.patient_ids = np.array(data["patientID"])

        self.global_context_in = full_data_input[:context_size]
        self.global_context_out = full_data_output[:context_size]

        if patient_ids is not None:
            mask = np.isin(self.patient_ids, patient_ids)
            self.data_input = full_data_input[mask]
            self.data_output = full_data_output[mask]
            self.patient_ids = self.patient_ids[mask]
        else:
            self.data_input = full_data_input
            self.data_output = full_data_output

    def __len__(self):
        return len(self.data_input)
    
    def __getitem__(self, idx):
        context_in = self.global_context_in
        context_out = self.global_context_out
        X = preprocess(self.data_input[idx])
        y = preprocess(self.data_output[idx])
        context_in, context_out = preprocess_context(context_in, context_out)
    

    
        return X.squeeze(), y.squeeze(), context_in, context_out



class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.output_dir = "Multi"
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




    def test_step(self, batch, batch_idx):
        import os
        import matplotlib.pyplot as plt

        def save_grayscale_image(image, folder, filename):
            import os
            import matplotlib.pyplot as plt

            if len(image.shape) == 3:
                image = image.mean(axis=0)

            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close()

        input_img, ground_truth_img, context_in, context_out = batch

        y_pred = self(input_img, context_in, context_out)
        ssim_val = ssim(y_pred, ground_truth_img).item()
        logging.info(ssim_val)
        self.log("test_ssim", ssim_val, prog_bar=True)
        input_img = input_img.squeeze().cpu().detach().numpy()
        ground_truth_img = ground_truth_img.squeeze().cpu().detach().numpy()
        y_pred = y_pred.squeeze().cpu().detach().numpy()

        if input_img.ndim == 3:
            input_img = input_img.mean(axis=0)
        if ground_truth_img.ndim == 3:
            ground_truth_img = ground_truth_img.mean(
                axis=0)
        if y_pred.ndim == 3:
            y_pred = y_pred.mean(axis=0)

        input_dir = os.path.join(self.output_dir, "input")
        ground_truth_dir = os.path.join(self.output_dir, "ground_truth")
        #output_dir = os.path.join(self.output_dir, "Generalized PA OmniNet")
        output_dir = os.path.join(self.output_dir, "Specific PA OmniNet")

        save_grayscale_image(input_img, input_dir, f"input_{batch_idx}.png")
        save_grayscale_image(ground_truth_img, ground_truth_dir,
                             f"ground_truth_{batch_idx}.png")
        #save_grayscale_image(y_pred, output_dir, f"generalized_pa_omninet_{batch_idx}.png")
        save_grayscale_image(y_pred, output_dir,
                              f"specific_pa_omninet_{batch_idx}.png")

        return {"batch_idx": batch_idx} 

if __name__ == "__main__":
    hparams = {
        "batch_size": 1,
        "learning_rate": 1e-4,
        "nb_levels": 4,
        "nb_inner_channels": 32,
        "nb_conv_layers_per_stage": 2,
        "data_slice_only": True,
        "max_epochs": 250,
    }
    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpoints/Neur4_patientSemi_final-epoch=49-val_loss=0.0011.ckpt"
    
    #file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_semicircle_RawBP.h5"
    #input_key = "sc,ss32_BP"
    #output_key = "sc_BP"
    
    #file_path = "/gpfs/home3/ostam/Datasets/oadat/swfd/SWFD_semicircle_RawBP-mini.h5"
    
    #file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/MSFD_multisegment_ss_RawBP.h5"
    #input_key ='ms,ss32_BP_w780'
    #output_key = 'ms_BP_w780'
    
    file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_multisegment_ss_RawBP.h5"
    input_key ='ms,ss32_BP'
    output_key = "ms_BP"
    batch_size = 1

    model = LightningModel(hparams)
    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpoints/final_model_semifinalpatient_early.pth"
    model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpoints/final_model_multifinalpatient_early.pth"
    #model_path = "/gpfs/home3/ostam/export_snellius/neuralizer/checkpoints/final_model_MFSDfinalpatient_early.pth"

    state_dict = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(state_dict)
    logging.info(f"Using model {model_path} on data {file_path}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(DEVICE)

    #model = LightningModel.load_from_checkpoint(model_path)
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    #model.eval()

    with h5py.File(file_path, 'r') as h5_file:
        patient_ids = h5_file['patientID'][:]
        total_patient_ids = sorted(set(map(int, patient_ids)))
    test_patient_ids = total_patient_ids[-2:]
    val_patient_ids = total_patient_ids[-4:-2]
    train_patient_ids = total_patient_ids[:-4]
    logging.info(f"Different patients: {total_patient_ids} \nTraining on patients {train_patient_ids}\nValidating on patients {val_patient_ids}\nTesting on patients       {test_patient_ids} ")

    logging.info(f"Training IDs: {train_patient_ids}")
    logging.info(f"Validation IDs: {val_patient_ids}")
    logging.info(f"Testing IDs: {test_patient_ids}")

    test_dataset = OADATDataloader(file_path, input_key=input_key,
                                   output_key=output_key, context_size=16,
                                   patient_ids=test_patient_ids)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=12, persistent_workers=True)

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=False
    )
    test_results = trainer.test(model, test_loader)
    logging.info(f"Test results: {test_results}")


