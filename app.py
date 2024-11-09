import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model (assuming the model file is in the same directory)
from tensorflow.keras.models import load_model
model = load_model('defect_detection_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(uploaded_img, target_size=(224, 224)):
    img = uploaded_img.resize(target_size)  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale the image to [0, 1]
    return img_array

# Streamlit app setup
st.title("Defect Detection Using DenseNet121")
st.write("Upload an image to predict if it's defective or non-defective.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    uploaded_img = Image.open(uploaded_file)
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(uploaded_img)
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    # Display the prediction result
    if prediction > 0.5:
        st.write(f"**Prediction:** Non-Defective")
        st.write(f"**Confidence:** {prediction[0][0] * 100:.2f}%")
    else:
        st.write(f"**Prediction:** Defective")
        st.write(f"**Confidence:** {100 - prediction[0][0] * 100:.2f}%")


