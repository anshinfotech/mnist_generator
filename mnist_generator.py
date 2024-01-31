import tensorflow as tf
from tensorflow import keras
from keras import models
import matplotlib.pyplot as plt
import numpy as np 
import streamlit as st
import cv2

st.set_page_config(page_title="MNIST Data Generator", layout="wide")
def mnist_page(confidence):
    gen = keras.models.load_model('mnist_generator.h5')
    desc = keras.models.load_model('mnist_discriminator.h5')

    st.title("MNIST Handwritten Digit Generator")
    itr = st.number_input("Select number of random images to be generated:", min_value=1, max_value=None, value=1, step=1)

    enhance = st.checkbox("Show enhanced image (may cause slowdown)", value=False)
    generate_button = st.button("Generate")



    if generate_button:
        while itr > 0:
            generated_image = gen.predict(np.random.normal(0, 1, (1, 100)))
            #convert into 0 to 255
            res = desc.predict(generated_image)
            if res > confidence:
                generated_image = generated_image * 127.5 + 127.5
                generated_image = generated_image.astype(np.uint8)
                if enhance:
                    generated_image = np.squeeze(generated_image)
                    generated_image = process(generated_image,200,0)
                st.image(generated_image)
                itr -= 1


def process(image,th,tl):
    image = cv2.pyrUp(image)
    #image = cv2.pyrUp(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] > th:
                image[i][j] = 255
            if image[i][j] < tl:
                image[i][j] = 0
    return image

def fashion_page(confidence):
    st.title("Fashion MNIST Image Generator")
    gen = keras.models.load_model('fashion_generator.h5')
    desc = keras.models.load_model('fashion_discriminator.h5')
    itr = st.number_input("Select number of random images to be generated:", min_value=1, max_value=None, value=1, step=1)
    enhance = st.checkbox("Show enhanced image (may cause slowdown)", value=False)
    generate_button = st.button("Generate")
    if generate_button:
        while itr > 0:
            generated_image = gen.predict(np.random.normal(0, 1, (1, 32*32)))
            res = desc.predict(generated_image)
            if res > confidence:
                generated_image = generated_image * 127.5 + 127.5
                generated_image = generated_image.astype(np.uint8)
                if enhance:
                    generated_image = np.squeeze(generated_image)
                    generated_image = process(generated_image,200,0)
                st.image(generated_image)
                itr -= 1


def homepage():
    st.write(
        """
        ## Welcome to the MNIST Data Generator!

        This application allows you to generate custom handwritten digits using a Generative Adversarial Network (GAN). 
        The GAN model has been trained on the MNIST dataset, which contains 28x28 grayscale images of handwritten digits (0-9).

        You can use this tool to:
        - Generate new handwritten digits similar to the ones in the MNIST dataset
        - Generate new images similar to the ones in the Fashion MNIST dataset

        """
    )  

    st.divider()
    st.write(""" 
        ### MNIST Data Generator:

        This GAN model has been trained on the MNIST dataset, which contains 28x28 grayscale images of handwritten digits (0-9).

        To use it, select the 'MNIST digits' page in the sidebar and experiment with the confidence value.

    """)

    st.divider()
    st.write(""" 
        ### Fashion MNIST Data Generator:

        This GAN model has been trained on the Fashion MNIST dataset, which contains 28x28 grayscale images of fashion MNIST images.
        To use it, select the 'Fashion MNIST' page in the sidebar and experiment with the confidence value.

    """)

    st.divider()
    st.write(""" 
        ### Confidence Value:
        The confidence value is used to determine if the generated image is similar to the original image. 
        The higher the confidence value, the more similar the images will be. 
        But slower will be the results.

    """)




st.title("MNIST Data Generator")
st.divider()
bar = st.sidebar
options = ('Home',"MNIST", "Fashion MNIST")
option = bar.selectbox("Select Page", options)
if option == "Fashion MNIST":
    confidence = bar.number_input("Set Confidence Threshold:", min_value=0.0, max_value=0.75, value=0.5, step=0.05)
    fashion_page(confidence)
    bar.warning('Still under development. The results may not be accurate.')
if option == "MNIST":
    confidence = bar.number_input("Set Confidence Threshold:", min_value=0.0, max_value=0.75, value=0.5, step=0.05)
    mnist_page(confidence)

if option == "Home":
    homepage()
    bar.text("Links:")
    bar.info("[Visit my GitHub account](https://github.com/Lackyjian)")
    bar.info("[Visit my LinkedIn account](https://www.linkedin.com/in/lakshay-jain-ab1895281/)")
    bar.info("[Visit my Kaggle account](https://www.kaggle.com/lakshayjain611)")
