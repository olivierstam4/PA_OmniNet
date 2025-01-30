import pytorch_lightning as pl
from PA_OmniNet.models.pairwise_conv_avg_model import PairwiseConvAvgModel
from PA_OmniNet.util.shapecheck import ShapeChecker
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


def preprocess_mat_Neuralizer(image, size=(256, 256)):
    image = (image - image.min()) / (image.max() - image.min())
    image = np.stack([image] * 3, axis=-1)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2).float()
    return image

def preprocess_batch_Neuralizer(images, size=(256, 256)):
    processed_images = []
    for image in images:
        processed_image = preprocess_mat_Neuralizer(image, size)
        processed_images.append(processed_image)
    return torch.cat(processed_images, dim=0)
def data_loader_formatter(filepath):
    data = np.load(filepath)
    data_in = data['arr_0']
    data_out = data['arr_1']
    return data_in, data_out
class CustomH5Dataset(Dataset):
    def __init__(self, data_in, data_out, context_size=16):
        self.context_size = context_size
        self.data_in = data_in
        self.data_out = data_out
        assert len(self.data_in) > context_size, "Insufficient data for the given context size."

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, idx):
        X = preprocess_mat_Neuralizer(self.data_in[idx])
        y = preprocess_mat_Neuralizer(self.data_out[idx])

        context_in = [self.data_in[(idx + 1 + i) % len(self.data_in)] for i in range(self.context_size)]
        context_out = [self.data_out[(idx + 1 + i) % len(self.data_out)] for i in range(self.context_size)]

        X_context = preprocess_batch_Neuralizer(context_in)
        y_context = preprocess_batch_Neuralizer(context_out)

        return X.squeeze(), y.squeeze(), X_context, y_context

class LightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = self._build_model()
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
        "batch_size": 32,
        "learning_rate": 1e-4,
        "nb_levels": 4,
        "nb_inner_channels": 32,
        "nb_conv_layers_per_stage": 2,
        "data_slice_only": True,
        "max_epochs": 10,
    }
    cuda_available = torch.cuda.is_available()
    current_device = torch.cuda.current_device() if cuda_available else None
    gpu_name = torch.cuda.get_device_name(current_device) if cuda_available else "No GPU"
    CONTEXT_SIZE = 4

    test_path_multi = "../../Datasets/3hz/Multi/test_set_256.npz"
    train_path_multi = "../../Datasets/3hz/Multi/train_set_256.npz"
    val_path_multi = "../../Datasets/3hz/Multi/valid_set_256.npz"

    test_path_single = "../../Datasets/3hz/Single/test_set_256.npz"
    train_path_single = "../../Datasets/3hz/Single/train_set_256.npz"
    val_path_single = "../../Datasets/3hz/Single/valid_set_256.npz"

    train_in, train_out = data_loader_formatter(train_path_multi)
    val_in, val_out = data_loader_formatter(val_path_multi)
    test_in, test_out = data_loader_formatter(test_path_multi)

    train_dataset = CustomH5Dataset(train_in, train_out, context_size=CONTEXT_SIZE)
    val_dataset = CustomH5Dataset(val_in, val_out, context_size=CONTEXT_SIZE)
    test_dataset = CustomH5Dataset(test_in, test_out, context_size=CONTEXT_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=4, persistent_workers=True)

    logging.info(f"Train dataset size: {len(train_dataset)} samples")
    logging.info(f"Validation dataset size: {len(val_dataset)} samples")

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
        filename='test_model_3hzMulti-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=hparams['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(model, train_loader, val_loader)
    model_path = 'checkpoints/final_model_3hzMulti.pth'
    torch.save(model.state_dict(), model_path)

    logging.info(
        f"Training started with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    trainer.fit(model, train_loader, val_loader)

    model_path = 'checkpoints/test_3hzMulti.pth'
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model parameters saved to {model_path}.")
    logging.info("Training complete.")
