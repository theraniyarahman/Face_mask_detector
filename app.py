import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configure Page Settings
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)

# Title and Description
st.title("😷 Face Mask Detection App")
st.write("Upload an image or use your webcam to check if a person is wearing a face mask.")

# Cache model loading so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("face_mask_detector_model.keras")

try:
    model = load_model()
    class_names = ['with_mask', 'without_mask']
except Exception as e:
    st.error(f"Error loading model. Make sure 'face_mask_detector_model.keras' is in the same directory. Details: {e}")
    st.stop()

# Choice of Input
option = st.radio("Choose Input Source:", ("Upload Image", "Use Webcam"))
image_file = None

if option == "Upload Image":
    image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
elif option == "Use Webcam":
    image_file = st.camera_input("Take a picture")

if image_file is not None:
    # Open and display image
    image = Image.open(image_file)
    st.image(image, caption="Input Image", use_container_width=True)
    
    # Preprocess
    # Convert image to RGB (in case of RGBA/Grayscale) and resize
    img_resized = image.convert("RGB").resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # Run inference
    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array, verbose=0)[0][0]
        
    # Determine results
    if prediction > 0.5:
        label = "No Mask Detected"
        confidence = prediction
        st.error(f"🚨 {label} ({confidence:.2%})")
    else:
        label = "Mask Detected"
        confidence = 1.0 - prediction
        st.success(f"✅ {label} ({confidence:.2%})")
