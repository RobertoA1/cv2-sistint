import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np

st.set_page_config(page_title="Capítulo 08: Frame Differencing", page_icon="🎥")
st.title('Capítulo 08: Frame Differencing')

scaling_factor = 1

class FrameDiffState:
    def __init__(self):
        self.prev_frame = None
        self.cur_frame = None

state = FrameDiffState()

def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
    return cv2.bitwise_and(diff_frames1, diff_frames2)

def frame_diff_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    next_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prev_frame = state.prev_frame
    cur_frame = state.cur_frame
    state.prev_frame = cur_frame
    state.cur_frame = next_frame
    if prev_frame is not None and cur_frame is not None:
        diff = frame_diff(prev_frame, cur_frame, next_frame)
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        return av.VideoFrame.from_ndarray(diff_color, format="bgr24")
    else:
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="frame_diff",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
    video_frame_callback=frame_diff_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
