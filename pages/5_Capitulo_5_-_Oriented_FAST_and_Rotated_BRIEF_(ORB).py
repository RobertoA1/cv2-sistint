import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np

st.set_page_config(page_title="Capítulo 05: ORB (Oriented FAST and Rotated BRIEF)", page_icon="🔎")
st.title('Capítulo 05: ORB (Oriented FAST and Rotated BRIEF)')

def orb_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    keypoints = orb.detect(gray_image, None)
    keypoints, descriptors = orb.compute(gray_image, keypoints)
    out_img = cv2.drawKeypoints(img, keypoints, img, color=(0,255,0))
    return av.VideoFrame.from_ndarray(out_img, format="bgr24")

webrtc_streamer(
    key="sistinteligentes5",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_frame_callback=orb_frame,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
