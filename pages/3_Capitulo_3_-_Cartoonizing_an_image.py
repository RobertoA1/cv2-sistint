import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np

st.set_page_config(page_title="Capítulo 03: Cartoonizing an Image", page_icon="🎨")
st.title('Capítulo 03: Cartoonizing an Image')

mode = st.radio('Modo de cartoonización:', ["Sin color (sketch)", "Con color"])
ksize = st.slider('Tamaño de kernel para bordes', 3, 11, 5, step=2)

def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 7)
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR)
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst


def cartoonize_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    sketch_mode = (mode == "Sin color (sketch)")
    cartoon = cartoonize_image(img, ksize=ksize, sketch_mode=sketch_mode)
    return av.VideoFrame.from_ndarray(cartoon, format="bgr24")

webrtc_streamer(
    key="sistinteligentes3",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_frame_callback=cartoonize_frame,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
