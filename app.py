import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- App Configuration ---
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

# --- Define Class Names ---
# These must match the exact alphabetical order from the training directory
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 
    'Pepper__bell___healthy', 
    'Potato___Early_blight', 
    'Potato___healthy', 
    'Potato___Late_blight', 
    'Tomato_Target_spot', 
    'Tomato_Tomato_mosaic_virus', 
    'Tomato_Tomato_YellowLeaf__Curl_Virus', 
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight', 
    'Tomato_healthy', 
    'Tomato__Late_blight', 
    'Tomato__Leaf_Mold', 
    'Tomato__Septoria_leaf_spot', 
    'Tomato__Spider_mites_Two_spotted_spider_mite',
]

# --- Load Model ---
# We use st.cache_resource so the model only loads once when the app starts
@st.cache_resource
def load_disease_model():
    model_path = 'plant_disease_model.keras'
    return tf.keras.models.load_model(model_path)

model = load_disease_model()

# --- UI Header ---
st.title("🌿 Plant Disease Classifier")
st.markdown("""
Upload an image of a pepper, potato, or tomato leaf, and the deep learning model will analyze it for signs of disease.
""")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", width='content')
    
    st.markdown("---")
    
    # Predict button
    if st.button("Analyze Leaf"):
        with st.spinner("Analyzing image..."):
            # 1. Preprocess the image
            # Resize to match the 128x128 input shape our CNN expects
            img = image.resize((128, 128)) 
            
            # Convert to numpy array and ensure it has 3 channels (RGB)
            img_array = np.array(img.convert('RGB')) 
            
            # Expand dimensions to create a batch of 1 (shape: 1, 128, 128, 3)
            img_batch = np.expand_dims(img_array, axis=0)
            
            # 2. Make Prediction
            # Note: We don't divide by 255 here because our CNN has a built-in Rescaling layer
            predictions = model.predict(img_batch)
            
            # 3. Process Results
            predicted_class_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_index]
            confidence = np.max(predictions[0]) * 100
            
            # 4. Display Results
            # Format the output to look cleaner
            formatted_class = predicted_class.replace('___', ' - ').replace('__', ' ').replace('_', ' ')
            
            st.success(f"**Prediction:** {formatted_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            
            # Add a visual warning if confidence is low
            if confidence < 75.0:
                st.warning("Confidence is somewhat low. Ensure the image is clear and focused on a single leaf.")