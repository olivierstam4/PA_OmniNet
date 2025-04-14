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


def preprocess_mouse(image):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.stack([image] * 3, axis=-1)
    image = torch.tensor(image).unsqueeze(0).permute(0, 3, 1, 2).float()
    return image


def preprocess_mouse_batch(images):
    processed_images = [preprocess_mouse(image) for image in images]
    return torch.cat(processed_images, dim=0)


def data_loader_form_mouse(in_filepath, out_filepath):
    with h5py.File(in_filepath, 'r') as infile:
        first_key = list(infile.keys())[0]
        data_in = infile[first_key][()]

    with h5py.File(out_filepath, 'r') as outfile:
        first_key = list(outfile.keys())[0]
        data_out = outfile[first_key][()]

    return data_in, data_out

class DataloaderMouse(Dataset):
    def __init__(self, data_in, data_out, context_size=16):
        self.context_size = context_size
        self.data_in = data_in
        self.data_out = data_out

        assert len(self.data_in) > context_size, "Insufficient data for the given context size."

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, idx):
        X = preprocess_mouse(self.data_in[idx])
        y = preprocess_mouse(self.data_out[idx])
        context_in = [self.data_in[(idx + 1 + i) % len(self.data_in)] for i in
                      range(self.context_size)]
        context_out = [self.data_out[(idx + 1 + i) % len(self.data_out)] for i
                       in range(self.context_size)]
        X_context = preprocess_mouse_batch(context_in)
        y_context = preprocess_mouse_batch(context_out)

        return X.squeeze(), y.squeeze(), X_context, y_context

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
        print(context_in.shape)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])

if __name__ == "__main__":
    hparams = {
        "batch_size": 1,
        "learning_rate": 1e-4,
        "nb_levels": 4,
        "nb_inner_channels": 32,
        "nb_conv_layers_per_stage": 2,
        "data_slice_only": True,
        "max_epochs": 50,
    }
    cuda_available = torch.cuda.is_available()
    current_device = torch.cuda.current_device() if cuda_available else None
    gpu_name = torch.cuda.get_device_name(current_device) if cuda_available else "No GPU"
    mice = ['../../Datasets/optoacousticsparse/mice_sparse16_recon_256.mat',
            '../../Datasets/optoacousticsparse/mice_sparse32_recon_256.mat',
            '../../Datasets/optoacousticsparse/mice_sparse128_recon_256.mat']
    v_phantom = [
        '../../Datasets/optoacousticsparse/v_phantom_sparse16_recon_256.mat',
        '../../Datasets/optoacousticsparse/v_phantom_sparse32_recon_256.mat',
        '../../Datasets/optoacousticsparse/v_phantom_full_recon_256.mat']

    logging.info(f"CUDA Available: {cuda_available}")
    logging.info(f"Current CUDA Device: {current_device if cuda_available else 'N/A'}")
    logging.info(f"GPU Name: {gpu_name}")
    CONTEXT_SIZE = 4

    input_file_path = mice[1]
    output_file_path = mice[2]
    data_in, data_out = data_loader_form_mouse(input_file_path,
                                               output_file_path)

    total_samples = data_in.shape[0]
    indices = np.arange(total_samples)
    train_end = int(0.7 * total_samples)
    val_end = train_end + int(0.2 * total_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_dataset = DataloaderMouse(data_in[train_indices], data_out[train_indices], context_size=CONTEXT_SIZE)
    val_dataset = DataloaderMouse(data_in[val_indices], data_out[val_indices], context_size=CONTEXT_SIZE)
    test_dataset = DataloaderMouse(data_in[test_indices], data_out[test_indices], context_size=CONTEXT_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=4, persistent_workers=True)

    logging.info(f"Train dataset size: {len(train_dataset)} samples")
    logging.info(f"Validation dataset size: {len(val_dataset)} samples")
    logging.info(f"Test dataset size: {len(test_dataset)} samples")

    logging.info(f"Train DataLoader batches: {len(train_loader)}")
    logging.info(f"Validation DataLoader batches: {len(val_loader)}")

    os.makedirs('checkpoints/', exist_ok=True)

    logger = TensorBoardLogger("lightning_logs", name="model_logs")

    model = LightningModel(hparams)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='test_model_mouse-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='test_model_mouse2-{epoch:02d}-{val_loss:.4f}',
        save_top_k=-1,
        every_n_epochs=5,
        mode='min',
    )


    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    model_path = 'checkpoints/final_model_mousetest.pth'
    torch.save(model.state_dict(), model_path)

    logging.info(
        f"Training started with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    trainer.fit(model, train_loader, val_loader)

    model_path = 'checkpoints/test_mouse.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model parameters saved to {model_path}.")
    logging.info("Training complete.")


    logging.info(
        "Launch TensorBoard with the command: tensorboard --logdir=lightning_logs/")
