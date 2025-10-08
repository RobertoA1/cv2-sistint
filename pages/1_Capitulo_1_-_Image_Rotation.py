import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Cap. 01: Rotación de imágenes",
    page_icon="🛞"
)

st.title('Capítulo 01: Rotación de imágenes')

if "rotacion" not in st.session_state:
    st.session_state["rotacion"] = 0

input = st.camera_input("Toma una foto", key="sistinteligentes1")

with st.container():
    rotacionSlider = st.slider('Rotación de la imagen', 0, 360, 0)

st.session_state["rotacion"] = rotacionSlider

def rotarImagen(img, rotacion):
    try:
        num_rows, num_cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), rotacion, 1)
        rotated = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        return rotated
    except Exception as e:
        print("Error en rotarImagen:", e)
        return img

if input is not None:
    img = Image.open(input)
    img = np.array(img)
    frame = rotarImagen(img, rotacionSlider)
    st.image(frame, caption="Imagen rotada")