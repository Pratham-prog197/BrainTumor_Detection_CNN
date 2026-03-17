import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(image):
    """
    Preprocess MRI image for model prediction
    """
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image / 255.0
    return image.reshape(1, IMG_SIZE, IMG_SIZE, 3)


def postprocess_mask(prediction, threshold=0.5):
    """
    Convert model output to binary tumor mask
    """
    mask = prediction[0].squeeze()
    mask = (mask > threshold).astype(np.uint8)
    return mask