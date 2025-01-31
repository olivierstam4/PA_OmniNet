import pytorch_lightning as pl
from models.pairwise_conv_avg_model import PairwiseConvAvgModel
from util.shapecheck import ShapeChecker
import torch
import torch.nn.functional as F
import h5py
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import os
import logging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
            conv_layers_per_stage=self.hparams.nb_conv_layers_per_stage,
        )

    def forward(self, target_in, context_in, context_out):
        sc = ShapeChecker()
        sc.check(target_in, "B C H W", C=3, H=256, W=256)
        sc.check(context_in, "B L C H W")
        sc.check(context_out, "B L C H W")

        y_pred = self.net(context_in, context_out, target_in)
        sc.check(y_pred, "B C H W")
        return y_pred

    def training_step(self, batch, batch_idx):
        target_in, y, context_in, context_out = batch
        y_pred = self(target_in, context_in, context_out)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_in, y, context_in, context_out = batch
        y_pred = self(target_in, context_in, context_out)
        val_loss = F.mse_loss(y_pred, y)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        target_in, y, context_in, context_out = batch
        y_pred = self(target_in, context_in, context_out)
        test_loss = F.mse_loss(y_pred, y)

        if batch_idx == 0:
            y_pred_norm = (y_pred - y_pred.min()) / (
                        y_pred.max() - y_pred.min())
            y_norm = (y - y.min()) / (y.max() - y.min())
            pred_grid = torchvision.utils.make_grid(
                y_pred_norm[:4], normalize=False, scale_each=True
            ).permute(1, 2, 0).squeeze().cpu().numpy()
            target_grid = torchvision.utils.make_grid(
                y_norm[:4], normalize=False, scale_each=True
            ).permute(1, 2, 0).squeeze().cpu().numpy()

            self.logger.experiment.add_image(
                'Test/Predictions', pred_grid, 0, dataformats="HWC"
            )
            self.logger.experiment.add_image(
                'Test/GroundTruth', target_grid, 0, dataformats="HWC"
            )

        self.log('test_loss', test_loss, prog_bar=True)
        return test_loss

    def on_train_end(self):
        logger_dir = self.logger.log_dir
        model_path = os.path.join(logger_dir, "model_final_patient.pth")

        torch.save(self.state_dict(), model_path)
        print(f"Model parameters saved to {model_path}")
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
cpu_count = os.cpu_count()
num_workers = cpu_count / 2
num_workers = int(num_workers)
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
    cuda_available = torch.cuda.is_available()
    current_device = torch.cuda.current_device() if cuda_available else None
    gpu_name = torch.cuda.get_device_name(current_device) if cuda_available else "No GPU"

    logging.info(f"CUDA Available: {cuda_available}")
    logging.info(f"Current CUDA Device: {current_device if cuda_available else 'N/A'}")
    logging.info(f"GPU Name: {gpu_name}")

    #file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_semicircle_RawBP.h5"
    #file_path = "/gpfs/home3/ostam/Datasets/oadat/swfd/SWFD_semicircle_RawBP-mini.h5"
    #file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/MSFD_multisegment_ss_RawBP.h5"
    #input_key ='ms,ss32_BP_w780'
    #output_key = 'ms_BP_w780'
    file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_multisegment_ss_RawBP.h5"
    input_key ='ms,ss32_BP'
    output_key = "ms_BP"
    #file_path = "/gpfs/scratch1/shared/tmp.zlixKwrks6/SWFD_multisegment_ss_RawBP.h5"
    #input_key ='sc,ss32_BP'
    #output_key = "sc_BP"
    with h5py.File(file_path, 'r') as h5_file:
        patient_ids = h5_file['patientID'][:]
        total_patient_ids = sorted(set(map(int, patient_ids)))

    test_patient_ids = total_patient_ids[-2:]
    val_patient_ids = total_patient_ids[-4:-2]
    train_patient_ids = total_patient_ids[:-4]

    train_dataset = OADATDataloader(file_path, input_key=input_key, output_key=output_key, patient_ids=train_patient_ids)
    val_dataset = OADATDataloader(file_path, input_key=input_key, output_key=output_key, patient_ids=val_patient_ids)
    test_dataset = OADATDataloader(file_path, input_key=input_key, output_key=output_key, patient_ids=test_patient_ids)

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=num_workers)

    logger = TensorBoardLogger("lightning_logs", name="model_logs")

    model = LightningModel(hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='Neur4_patientmulti_final-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        every_n_epochs=5,
        mode='min',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=30,
        verbose=True,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
      monitor='val_loss',
      dirpath='checkpoints/',
      filename='Best_modelmulti-{epoch:02d}-{val_loss:.4f}',
      save_top_k=1,
      mode='min',
    )


    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    logging.info(f"Training started with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    trainer.fit(model, train_loader, val_loader)

    model_path = 'checkpoints/final_model_multifinalpatient_early.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model parameters saved to {model_path}.")
    logging.info("Training complete.")

    logging.info("Starting test phase...")
    test_results = trainer.test(model, test_loader)
    logging.info(f"Test results: {test_results}")


