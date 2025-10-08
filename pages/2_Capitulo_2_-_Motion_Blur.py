import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Capítulo 02: Motion Blur",
    page_icon="�"
)

st.title('Capítulo 02: Motion Blur')

input = st.camera_input("Toma una foto para aplicar Motion Blur", key="motionblur")

size = st.slider('Tamaño del kernel de desenfoque', 1, 50, 25)

def motionBlur(img, size):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    output = cv2.filter2D(img, -1, kernel_motion_blur)
    return output

if input is not None:
    img = Image.open(input)
    img = np.array(img)
    blurred = motionBlur(img, size)
    st.image(blurred, caption=f"Motion Blur (kernel={size})")