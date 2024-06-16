import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('handwritten_digit_classifier.h5')


def predict_class(img):
    if img is None:
        st.error("No image provided.")
        return None

    img = img.resize((28, 28)).convert('L')
    img = np.array(img)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    return str(predicted_class)

st.markdown(
    """
    <style>
    h2{
        text-align: center;
        font-family: 'poppins', sans-serif;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 2px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #000;
        margin-top: 2rem;
    }
    .copyright {
        text-align: center;
        font-size: 0.8rem;
        color: #000;
        margin-top: 1rem;
    }
    .footer a{
        font-family: 'poppins', sans-serif;
        font-size: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main"> <h2>Handwritten Digit Classifier</h2>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))
    st.image(image, caption='Uploaded Image.', use_column_width=False)

    if st.button("Predict"):
        digit = predict_class(image)
        if digit:
            st.success(f"Predicted Digit: {digit}")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('''
    <div class="footer">
        Developed by Prabhat Kumar Raj
        <br>
        <a href="https://github.com/Prabhat-2101/Handwritten-Digit-Recognition-using-deep-learning" target="_blank">View on GitHub</a>
    </div>
    <div class="copyright">
        &copy; 2024 All rights reserved @digitdetectorai
    </div>
    ''', unsafe_allow_html=True)