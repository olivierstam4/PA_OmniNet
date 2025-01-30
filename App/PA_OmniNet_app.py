## cd to file dir and run command: > streamlit run PA_OmniNet_app.py

import os
import torch
import numpy as np
from PIL import Image
from PA_OmniNet_backend import process_images
import streamlit as st

st.title("Image Visualization Tool")
st.subheader("Please upload a single input image")
input_image = st.file_uploader("Accepted formats are: PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])

st.write("Ensure that context images (in and out) are uploaded in matching pairs and same amount.")
st.subheader("Upload Context Images")
context_in_files = st.file_uploader("Context-In Images",type=["png", "jpg", "jpeg"], accept_multiple_files=True)
context_out_files = st.file_uploader("Context-Out Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

st.write("A context size of 16 is recommended for optimal results.")
if context_in_files and context_out_files:
    max_context = min(len(context_in_files), len(context_out_files))
    context_size = st.slider("Select the number of context images", 1, max_context, 16)
else:
    context_size = 0
if input_image and context_size > 0:

    input_image_path = "temp_input.png"
    with open(input_image_path, "wb") as f:
        f.write(input_image.read())

    context_in_dir = "temp_context_in"
    os.makedirs(context_in_dir, exist_ok=True)
    for i, file in enumerate(context_in_files[:context_size]):
        with open(os.path.join(context_in_dir, f"context_in_{i}.png"), "wb") as f:
            f.write(file.read())

    context_out_dir = "temp_context_out"
    os.makedirs(context_out_dir, exist_ok=True)
    for i, file in enumerate(context_out_files[:context_size]):
        with open(os.path.join(context_out_dir, f"context_out_{i}.png"), "wb") as f:
            f.write(file.read())

    output_image = process_images(input_image_path, context_in_dir, context_out_dir, context_size)

    st.subheader("Output Image:")
    st.image(output_image, use_container_width=True, clamp=True, channels="RGB")

    os.remove(input_image_path)
    for temp_dir in [context_in_dir, context_out_dir]:
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
else:
    st.warning("Please upload all necessary images to proceed.")
