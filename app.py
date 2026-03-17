import streamlit as st
import numpy as np
import cv2
import gdown
import os
import matplotlib.pyplot as plt
import tensorflow.lite as tflite

IMG_SIZE = 128
MODEL_URL = "https://drive.google.com/uc?id=1Z0uBVDwbfjMGC7BAxeLkFnK03n9-PDCL"
MODEL_PATH = "brain_tumor_model.tflite"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

st.title("🧠 Brain Tumor Segmentation (U-Net + TFLite)")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.subheader("Uploaded MRI")
    st.image(img)

    input_img = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], input_img)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])
    mask = (pred[0].squeeze() > 0.5).astype(np.uint8)

    st.subheader("Predicted Tumor Mask")

    fig, ax = plt.subplots()
    ax.imshow(mask, cmap="gray")
    ax.axis("off")
    st.pyplot(fig)