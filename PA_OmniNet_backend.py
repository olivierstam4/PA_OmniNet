import os
from PIL import Image
from App_model import LightningModel
from helper_functions import *
import cv2

model_weights = 'weights/PAOmni/OADAT/Alpha/PA_OmniNet_Generalized.ckpt'
hparams = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "nb_levels": 4,
    "nb_inner_channels": 32,
    "nb_conv_layers_per_stage": 2,
    "data_slice_only": True,
    "max_epochs": 250,
}

App_model = LightningModel.load_from_checkpoint(checkpoint_path=model_weights, map_location=torch.device('cpu'))
App_model.eval()
torch.set_grad_enabled(False)

def preprocess_app(image, size=(256, 256)):
    image = np.array(image)
    image = (image - image.min()) / (image.max() - image.min())
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = np.stack([image] * 3, axis=-1)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2).float()
    return image

def preprocess_context_app(directory, context_size):
    images = []
    for i, file in enumerate(sorted(os.listdir(directory))):
        if i >= context_size:
            break
        if file.endswith(".png"):
            img_path = os.path.join(directory, file)
            image = Image.open(img_path).convert('L')
            image = np.array(image)
            image = preprocess_app(image)
            images.append(image)
    return torch.cat(images, dim=0)

def process_images(input_image_path, context_in_dir, context_out_dir, context_size):
    input_image = Image.open(input_image_path).convert('L')
    input_image = np.array(input_image)
    input_image_tensor = preprocess_app(input_image).unsqueeze(0)
    context_in_tensor = preprocess_context_app(context_in_dir,
                                               context_size).unsqueeze(0)
    context_out_tensor = preprocess_context_app(context_out_dir,
                                                context_size).unsqueeze(0)
    output_tensor = App_model.forward(input_image_tensor, context_in_tensor, context_out_tensor)
    output_image = output_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    return output_image
