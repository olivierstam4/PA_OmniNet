import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from unet1 import UNet
import logging
from tqdm import tqdm

logging.basicConfig(
    filename="training_log.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger()
logger.addHandler(console_handler)

TRAIN_DATASET = "../Datasets/3hz/Single/train_set_256.npz"
VALID_DATASET = "../Datasets/3hz/Single/valid_set_256.npz"
TEST_DATASET = "../Datasets/3hz/Single/test_set_256.npz"

GAMMA = 0.98
MOMENTUM = 0.9
STEP_SIZE = 1
LEARNING_RATE = 1e-4
EPOCHS = 250
LOSS_MUL = 1e4
num_patches = 100
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 1
BATCH_SIZE = 1
START_FILTERS = 32
MODEL_SAVE_PATH = "unet_model_3hz_Single.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Dataloader3hz(Dataset):
    def __init__(self, file_path, normalize=True):
        self.file_path = file_path
        self.normalize = normalize
        data = np.load(file_path)
        self.inputs = data['arr_0']
        self.outputs = data['arr_1']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]

        if self.normalize:
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)

        return x_tensor, y_tensor

if __name__ == '__main__':
    EARLY_STOPPING_PATIENCE = 50
    best_val_loss = float('inf')
    early_stop_counter = 0

    train_dataset = Dataloader3hz(TRAIN_DATASET, normalize=True)
    val_dataset = Dataloader3hz(VALID_DATASET, normalize=True)
    test_dataset = Dataloader3hz(TEST_DATASET, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"Train dataset size: {len(train_dataset)} samples")
    logger.info(f"Validation dataset size: {len(val_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)} samples")

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
            torch.save(model.state_dict(),
                       "best_model_3hz_Single.pth")
            logger.info(f"Validation loss improved. Model saved as best_model_3hz.pth")
        else:
            early_stop_counter += 1
            logger.info(f"No improvement in validation loss for {early_stop_counter} consecutive epochs.")

        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered. Stopping training.")
            break

        if (epoch + 1) % 50 == 0:
            checkpoint_path = os.path.join("checkpoints", f"Unet_3hz_Single{epoch + 1}.ckpt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Final model saved: {MODEL_SAVE_PATH}")
