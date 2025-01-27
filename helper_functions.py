import h5py
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2


def get_info(file_path):

    keys_list = []
    types_list = []

    try:
        with h5py.File(file_path, 'r') as h5_file:
            keys = list(h5_file.keys())

            for key in keys:
                data = h5_file[key]
                keys_list.append(key)
                data_type = (key, data.dtype)
                types_list.append(data_type)

    except Exception as e:
        print(f"Error opening file: {e}")

    return keys_list, types_list


def load_h5_data(file_path):

    data_dict = {}

    try:
        with h5py.File(file_path, 'r') as h5_file:
            for key in h5_file.keys():
                data = h5_file[key][...]
                data_dict[key] = np.array(data)

    except Exception as e:
        print(f"Error loading file: {e}")

    return data_dict


def useful_oadat(file_path):
    keys, types = get_info(file_path)
    filtered_keys = {
        key for key, dtype in types
        if "_raw" not in key and np.issubdtype(dtype, np.floating)
    }
    return filtered_keys


def load_images(file_name):
    with h5py.File(file_name, 'r') as f:
        data = list(f.values())[0][:]
    return data


def show_image(image, name='Grayscale Image'):
    plt.imshow(image, cmap='gray')
    plt.colorbar()
    plt.title(name)
    plt.show()


def show_image_clean(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def show_filtered_oadat(filepath, itteration=0):
    filtered_keys = useful_oadat(filepath)
    try:
        with h5py.File(filepath, 'r') as h5_file:
            for key in filtered_keys:
                show_image(h5_file[key][itteration], key)
    except Exception as e:
        print(f"Error opening or processing file: {e}")


def show_context(context_in, context_out):
    num_images_1 = len(context_in)
    num_images_2 = len(context_out)

    total_images = max(num_images_1, num_images_2)

    fig, axes = plt.subplots(nrows=2, ncols=total_images, figsize=(15, 6))

    for i in range(total_images):
        if i < num_images_1:
            axes[0, i].imshow(context_in[i], cmap='gray')
            axes[0, i].axis('off')
        else:
            axes[0, i].axis('off')

    for i in range(total_images):
        if i < num_images_2:
            axes[1, i].imshow(context_out[i], cmap='gray')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()


def data_loader_formatter(filepath, sparse_key, groundtruth_key):
    data = load_h5_data(filepath)
    sparse_data = data[sparse_key]
    full_data = data[groundtruth_key]
    return sparse_data, full_data


def creating_context(sparse_data, full_data, idx=0, context_size=4,
                     random=False):
    if len(sparse_data) != len(full_data):
        raise ValueError('The length of sparse and full data must be equal.')
    if random:
        idx_used = np.random.randint(0,
                                     high=(len(sparse_data) - 1 - context_size))
    else:
        idx_used = idx
    X = sparse_data[idx_used]
    y = full_data[idx_used]
    context_in = sparse_data[idx_used + 1: idx_used + 1 + context_size]
    context_out = full_data[idx_used + 1: idx_used + 1 + context_size]
    return X, y, context_in, context_out


def preprocess(image, size=(192, 192)):
    image = (image - image.min()) / (image.max() - image.min())
    # image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = np.stack([image] * 3, axis=-1)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2).float()
    return image


def preprocess_mat(image, size=(256, 256)):
    image = (image - image.min()) / (image.max() - image.min())
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = np.stack([image] * 3, axis=-1)
    image = torch.tensor(image)
    image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2).float()
    return image


def preprocess_context_mat(context_in, context_out):
    context_in_preprocessed = np.array([preprocess_mat(c) for c in context_in])
    context_in_preprocessed = (
        torch.tensor(context_in_preprocessed).permute(1, 0, 2, 3, 4))
    context_out_preprocessed = np.array(
        [preprocess_mat(c) for c in context_out])
    context_out_preprocessed = (
        torch.tensor(context_out_preprocessed).permute(1, 0, 2, 3, 4))
    return context_in_preprocessed, context_out_preprocessed


def preprocess_context(context_in, context_out):
    context_in_preprocessed = np.array([preprocess(c) for c in context_in])
    context_in_preprocessed = (
        torch.tensor(context_in_preprocessed).permute(1, 0, 2, 3, 4))
    context_out_preprocessed = np.array([preprocess(c) for c in context_out])
    context_out_preprocessed = (
        torch.tensor(context_out_preprocessed).permute(1, 0, 2, 3, 4))
    return context_in_preprocessed, context_out_preprocessed


def back_to_img(img):
    return img.squeeze(0).permute(1, 2, 0)


def data_loader_formatter_mat(in_filepath, out_filepath):
    with h5py.File(in_filepath, 'r') as infile:
        first_key = list(infile.keys())[0]
        data_in = infile[first_key][()]

    with h5py.File(out_filepath, 'r') as outfile:
        first_key = list(outfile.keys())[0]
        data_out = outfile[first_key][()]

    return data_in, data_out


def creating_context_mat(sparse_data, full_data, idx=0, context_size=4,
                         random=False):
    if len(sparse_data) != len(full_data):
        raise ValueError('The length of sparse and full data must be equal.')
    if random:
        idx_used = np.random.randint(0,
                                     high=(len(sparse_data) - 1 - context_size))
    else:
        idx_used = idx
    X = sparse_data[idx_used]
    y = full_data[idx_used]
    context_in = sparse_data[idx_used + 1: idx_used + 1 + context_size]
    context_out = full_data[idx_used + 1: idx_used + 1 + context_size]
    return X, y, context_in, context_out
