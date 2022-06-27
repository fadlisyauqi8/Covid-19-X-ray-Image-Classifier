import pickle
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from tensorflow import keras



st.set_page_config(layout="centered", page_icon="💀", page_title="Covid-19 X-ray Image Classifier")


# Load model
model = tf.keras.models.load_model("model_imp_final.h5")
img = None

LABEL = "Covid" , "Normal" , "Viral Pneumonia"


# Setting up the model variables
img_height= 220 # Height of image
img_width= 220 # Width of image

# Function Prediction
def load_image(img):
    pred = Image.open(img).resize(size=(img_height, img_width)) # Resize Images
    pred = np.array(pred) # Convert to numpy array
    pred = tf.expand_dims(pred, 0)
    #pred = pred/255. # Rescale Images
    res = model.predict(pred)
    res = tf.keras.layers.Softmax()(res)

    return np.argmax(res, axis = 1)[0],res

# Homepage
st.title("Covid-19 X-ray Image Classification")
st.subheader('This apps will help user to detect and classify X-ray images')
# Image Header
image = Image.open('HeaderFront.jpg')
st.image(image, use_column_width = True, caption='Covid-19 X-ray Image Classifier')

# Upload image section
st.subheader('Upload X-ray Image')
prediction = st.file_uploader("Upload a X-ray image to be used for classification", type=["jpg", "png", "jpeg"])
if prediction is not None: 
    st.image(prediction, use_column_width='auto')
    btn = st.button('Predict')
    if btn:
        pred = load_image(prediction)
        if pred :
            st.success('The image is {}'.format(LABEL[pred[0]]))
        else:
            st.error('Error')



            #st.write('The probability of the image is {}'.format(pred[1][pred[0]]))
            #st.markdown(f"""  {[pred]}  """)
        










