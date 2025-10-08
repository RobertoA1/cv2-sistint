import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Capítulo 06: Seam Carving", page_icon="🖼️")
st.title('Capítulo 06: Seam Carving')

uploaded_file = st.file_uploader("Sube una imagen para expandir", type=["jpg", "jpeg", "png"])
num_seams = st.slider('Número de seams a agregar', 1, 50, 10)

def compute_energy_matrix(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(sobelx) + np.abs(sobely)
    return energy

def find_vertical_seam(img, energy):
    rows, cols = energy.shape
    seam = np.zeros(rows, dtype=np.int32)
    cost = energy.copy()
    for i in range(1, rows):
        for j in range(cols):
            min_pre = cost[i-1, j]
            if j > 0:
                min_pre = min(min_pre, cost[i-1, j-1])
            if j < cols-1:
                min_pre = min(min_pre, cost[i-1, j+1])
            cost[i, j] += min_pre
    seam[-1] = np.argmin(cost[-1])
    for i in range(rows-2, -1, -1):
        prev_x = seam[i+1]
        min_x = prev_x
        if prev_x > 0 and cost[i, prev_x-1] < cost[i, min_x]:
            min_x = prev_x-1
        if prev_x < cols-1 and cost[i, prev_x+1] < cost[i, min_x]:
            min_x = prev_x+1
        seam[i] = min_x
    return seam

def overlay_vertical_seam(img, seam):
    img_overlay = img.copy()
    for row in range(img.shape[0]):
        col = int(seam[row])
        if 0 <= col < img.shape[1]:
            img_overlay[row, col] = [0,0,255]
    return img_overlay

def add_vertical_seam(img, seam, num_iter):
    seam = seam + num_iter
    rows, cols = img.shape[:2]
    zero_col_mat = np.zeros((rows,1,3), dtype=np.uint8)
    img_extended = np.hstack((img, zero_col_mat))
    for row in range(rows):
        for col in range(cols, int(seam[row]), -1):
            img_extended[row, col] = img[row, col-1]
        for i in range(3):
            v1 = img_extended[row, int(seam[row])-1, i]
            v2 = img_extended[row, int(seam[row])+1, i]
            img_extended[row, int(seam[row]), i] = (int(v1)+int(v2))/2
    return img_extended

if uploaded_file is not None:
    img_input = Image.open(uploaded_file)
    img_input = np.array(img_input)
    if img_input.ndim == 3 and img_input.shape[2] == 4:
        img_input = cv2.cvtColor(img_input, cv2.COLOR_RGBA2BGR)
    img = np.copy(img_input)
    img_output = np.copy(img_input)
    img_overlay_seam = np.copy(img_input)
    energy = compute_energy_matrix(img)
    for i in range(num_seams):
        seam = find_vertical_seam(img, energy)
        img_overlay_seam = overlay_vertical_seam(img_overlay_seam, seam)
        img_output = add_vertical_seam(img_output, seam, i)
        img = np.copy(img_output)
        energy = compute_energy_matrix(img)
    st.image(img_input, caption="Imagen original")
    st.image(img_overlay_seam, caption="Seams agregados (en rojo)")
    st.image(img_output, caption="Imagen expandida")