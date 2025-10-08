import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Capítulo 07: Watershed Algorithm", page_icon="🌊")
st.title('Capítulo 07: Watershed Algorithm')

uploaded_file = st.file_uploader("Sube una imagen para segmentar con Watershed", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_input = Image.open(uploaded_file)
    img_input = np.array(img_input)
    if img_input.ndim == 3 and img_input.shape[2] == 4:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_RGBA2BGR)
    img = np.copy(img_input)
    st.image(img, caption="Imagen original")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0

    img_watershed = img.copy()
    cv2.watershed(img_watershed, markers)
    img_watershed[markers == -1] = [255,0,0]
    st.image(img_watershed, caption="Segmentación Watershed (bordes en rojo)")
