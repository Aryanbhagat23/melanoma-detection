import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

st.set_page_config(page_title="Melanoma Detector", page_icon="🔬", layout="centered")
st.title("🔬 Melanoma Detection System")
st.caption("Research Tool — Not for clinical use")

MODEL_PATH = "best_melanoma_model.keras"

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    if img_array.ndim == 3 and img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

if not os.path.exists(MODEL_PATH):
    st.error("Model file 'best_melanoma_model.keras' not found.")
    st.stop()

with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()

# Use dummy prediction to detect output shape — works on all Keras versions
dummy = preprocess_input(np.zeros((1, 224, 224, 3), dtype=np.float32))
test_out = model.predict(dummy, verbose=0)
output_units = int(test_out.shape[-1])

st.success(f"✅ Model loaded | Output units: {output_units} | Dummy output: {np.round(test_out, 4)}")

st.divider()
uploaded_file = st.file_uploader("Upload a dermoscopy image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True, caption="Uploaded Image")

    with st.spinner("Analyzing..."):
        processed = preprocess_image(img)
        raw_output = model.predict(processed, verbose=0)

    st.caption(f"🔧 Debug — Raw output: {np.round(raw_output, 4)} | Shape: {raw_output.shape}")

    if output_units == 1:
        # Sigmoid single output — value is probability of malignant
        prob_malignant = float(raw_output[0][0])
        prob_benign = 1.0 - prob_malignant
    else:
        # Softmax two outputs — apply softmax then read index 1 as malignant
        probs = tf.nn.softmax(raw_output[0]).numpy()
        prob_benign = float(probs[0])
        prob_malignant = float(probs[1])

    st.divider()
    st.write("### Result")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Malignant", f"{prob_malignant*100:.1f}%")
    with col2:
        st.metric("Benign", f"{prob_benign*100:.1f}%")

    st.progress(float(prob_malignant), text="Malignant risk")
    st.divider()

    if prob_malignant >= 0.5:
        st.error(f"🚨 MELANOMA DETECTED — {prob_malignant*100:.1f}% confidence")
        st.warning("Please consult a dermatologist immediately.")
    else:
        st.success(f"✅ BENIGN — {prob_benign*100:.1f}% confidence")
        st.info("Low risk. Regular skin checks still recommended.")

    st.divider()
    st.caption("ResNet50 Two-Stage Fine-Tuned | Accuracy 88.34% | Sensitivity 87.56% | AUC-ROC 0.9559")
    st.caption("⚠️ Research only. Not a substitute for professional medical diagnosis.")