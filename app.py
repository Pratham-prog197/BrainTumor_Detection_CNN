import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gdown
import os
from utils import preprocess_image, postprocess_mask

# =====================
# CONFIG
# =====================
IMG_SIZE = 128
MODEL_URL = "https://drive.google.com/uc?id=1rq6N83xcAzmxYOvMGQzkMOO9W6gSZ3AB"
MODEL_PATH = "brain_tumor_unet.h5"

# =====================
# DOWNLOAD MODEL (ONCE)
# =====================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# =====================
# PREPROCESS FUNCTION
# =====================
def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image.reshape(1, IMG_SIZE, IMG_SIZE, 3)

# =====================
# STREAMLIT UI
# =====================
st.title("🧠 Brain Tumor Segmentation using U-Net")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("Uploaded MRI")
    st.image(img, use_column_width=True)

input_img = preprocess_image(img)
pred = model.predict(input_img)
mask = postprocess_mask(pred)

    st.subheader("Predicted Tumor Mask")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    st.pyplot(plt)