import streamlit as st
from PIL import Image

from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import io

def inference_model(model_path='./best_weights.h5', image):
    model_ = load_model(model_path)
    classes = ['dry_asphalt_smooth','dry_concrete_smooth','dry_gravel']

    answer = classes[np.argmax(model_.predict(image))]
    return answer


def main():
    
    st.title("AlexNet-inspired Autoencoder: Image Classification Project")

    # Upload CSV file  
    image_path = st.file_uploader("Choose a image file", type=["jpg"])
    if image_path is not None:
        # uploaded_file = st.file_uploader("Upload Image")
        image = Image.open(image_path)
        image = np.array(image)
        
        image = cv2.resize(image, (224, 224))
        st.image(image, caption='Image')
        image = np.expand_dims(image, axis = 0)
        st.sidebar.title("Upload Model File")
        uploaded_model_file = st.sidebar.file_uploader("Choose a Model file", type=["h5"])
        if uploaded_model_file is not None:
            # Now you can use the 'model' variable for predictions or explanations
            class_predicted = inference_model(uploaded_model_file, image)
            st.write('Predicted class is ',class_predicted)
    # print(answer)
if __name__ == "__main__":
    main()
