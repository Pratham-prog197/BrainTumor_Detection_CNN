import streamlit as st
import numpy as np
import os
import requests
from PIL import Image
import tensorflow.lite as tflite

IMG_SIZE = 128
MODEL_URL = "https://drive.google.com/uc?export=download&id=1Z0uBVDwbfjMGC7BAxeLkFnK03n9-PDCL"
MODEL_PATH = "brain_tumor_model.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

@st.cache_resource
def load_model():
    download_model()
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

st.title("🧠 Brain Tumor Segmentation (U-Net + TFLite)")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded MRI")
    st.image(img)

    input_img = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])
    mask = (pred[0].squeeze() > 0.5).astype(np.uint8) * 255

    st.subheader("Predicted Tumor Mask")
    st.image(mask)