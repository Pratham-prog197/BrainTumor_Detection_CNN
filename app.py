import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os
from utils import preprocess_image, postprocess_mask

IMG_SIZE = 128
MODEL_URL = "https://drive.google.com/uc?id=1rq6N83xcAzmxYOvMGQzkMOO9W6gSZ3AB"
MODEL_PATH = "brain_tumor_unet.h5"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model


model = load_model()

st.title("🧠 Brain Tumor Segmentation using U-Net")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("Uploaded MRI")
    st.image(img, use_column_width=True)

    input_img = preprocess_image(img)

    with st.spinner("Predicting Tumor Region..."):
        pred = model.predict(input_img)

    mask = postprocess_mask(pred)

    st.subheader("Predicted Tumor Mask")

    fig, ax = plt.subplots()
    ax.imshow(mask, cmap="gray")
    ax.axis("off")

    st.pyplot(fig)