import os
import json
import numpy as np
from PIL import Image

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense,
    Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.optimizers import Adam


# =============================================================================
# Configuration
# =============================================================================
IMG_SIZE = 128  # must match training
NUM_CLASSES = 5

WEIGHTS_PATH = r"D:\AI\wbc_cnn_weights.h5"
CLASS_INDICES_PATH = r"D:\AI\class_indices.json"


# =============================================================================
# Rebuild the CNN architecture EXACTLY as in training
# =============================================================================
def build_cnn_model():
    model = Sequential([
        # Block 1
        Conv2D(64, (3, 3), activation="relu", padding="same",
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),

        # Block 4
        Conv2D(512, (3, 3), activation="relu", padding="same"),
        Conv2D(512, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(1024, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# Load model weights and class indices
# =============================================================================
@st.cache_resource
def load_model_and_labels():
    model = build_cnn_model()

    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"Weights file not found at: {WEIGHTS_PATH}")
        st.stop()

    try:
        model.load_weights(WEIGHTS_PATH)
    except Exception as e:
        st.error(f"Error loading weights:\n{e}")
        st.stop()

    if not os.path.exists(CLASS_INDICES_PATH):
        st.error(f"class_indices.json not found at: {CLASS_INDICES_PATH}")
        st.stop()

    try:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
    except Exception as e:
        st.error(f"Error reading class_indices.json:\n{e}")
        st.stop()

    return model, class_indices


model, class_indices = load_model_and_labels()


# =============================================================================
# Preprocessing
# =============================================================================
def preprocess_image(pil_image, target_size=(IMG_SIZE, IMG_SIZE)):
    img = pil_image.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# =============================================================================
# Prediction
# =============================================================================
def predict_image(model, pil_img):
    x = preprocess_image(pil_img)
    preds = model.predict(x)
    idx = int(np.argmax(preds))
    conf = float(preds[0][idx])
    label = class_indices.get(str(idx), f"Class {idx}")
    return label, conf


# =============================================================================
# Streamlit UI
# =============================================================================
st.title("White Blood Cell Classifier")

uploaded_image = st.file_uploader(
    "Upload a microscope image (JPG, JPEG, PNG):",
    type=["jpg", "jpeg", "png"],
)

if uploaded_image is not None:
    pil_image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image.resize((200, 200)), caption="Input Image")

    with col2:
        st.subheader("Prediction")
        if st.button("Classify"):
            label, conf = predict_image(model, pil_image)
            st.write(f"Predicted Class: {label}")
            st.write(f"Confidence: {conf * 100:.2f}%")
else:
    st.info("Upload an image to classify.")
