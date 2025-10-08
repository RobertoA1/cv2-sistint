import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np

st.set_page_config(page_title="Capítulo 09: Feature-based Tracking", page_icon="🟢")
st.title('Capítulo 09: Feature-based Tracking')

scaling_factor = 1
num_frames_to_track = st.slider('Frames a rastrear por punto', 5, 30, 15)
num_frames_jump = st.slider('Frames entre detección de nuevos puntos', 1, 10, 5)

tracking_params = dict(winSize=(15, 15), maxLevel=2,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Estado global para tracking
class TrackingState:
    def __init__(self):
        self.tracking_paths = []
        self.prev_gray = None
        self.frame_index = 0
state = TrackingState()

# Funciones de tracking

def compute_feature_points(tracking_paths, prev_img, current_img):
    feature_points = [tp[-1] for tp in tracking_paths]
    feature_points_0 = np.float32(feature_points).reshape(-1, 1, 2)
    feature_points_1, status_1, err_1 = cv2.calcOpticalFlowPyrLK(prev_img, current_img, feature_points_0, None, **tracking_params)
    feature_points_0_rev, status_2, err_2 = cv2.calcOpticalFlowPyrLK(current_img, prev_img, feature_points_1, None, **tracking_params)
    diff_feature_points = abs(feature_points_0 - feature_points_0_rev).reshape(-1, 2).max(-1)
    good_points = diff_feature_points < 1
    return feature_points_1.reshape(-1, 2), good_points

def calculate_region_of_interest(frame, tracking_paths):
    mask = np.zeros_like(frame)
    mask[:] = 255
    for x, y in [np.int32(tp[-1]) for tp in tracking_paths]:
        cv2.circle(mask, (x, y), 6, 0, -1)
    return mask

def add_tracking_paths(frame, tracking_paths):
    mask = calculate_region_of_interest(frame, tracking_paths)
    feature_points = cv2.goodFeaturesToTrack(frame, mask=mask, maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
    if feature_points is not None:
        for x, y in np.float32(feature_points).reshape(-1, 2):
            tracking_paths.append([(x, y)])

# Callback para el video

def tracking_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    frame = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    output_img = frame.copy()
    if len(state.tracking_paths) > 0 and state.prev_gray is not None:
        prev_img, current_img = state.prev_gray, frame_gray
        feature_points, good_points = compute_feature_points(state.tracking_paths, prev_img, current_img)
        new_tracking_paths = []
        for tp, (x, y), good_points_flag in zip(state.tracking_paths, feature_points, good_points):
            if not good_points_flag:
                continue
            tp.append((x, y))
            if len(tp) > num_frames_to_track:
                del tp[0]
            new_tracking_paths.append(tp)
            cv2.circle(output_img, (int(x), int(y)), 3, (0, 255, 0), -1)
        state.tracking_paths = new_tracking_paths
        point_paths = [np.int32(tp) for tp in state.tracking_paths]
        cv2.polylines(output_img, point_paths, False, (0, 150, 0))
    if state.frame_index % num_frames_jump == 0:
        add_tracking_paths(frame_gray, state.tracking_paths)
    state.frame_index += 1
    state.prev_gray = frame_gray
    return av.VideoFrame.from_ndarray(output_img, format="bgr24")

webrtc_streamer(
    key="feature_tracking",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=False,
    video_frame_callback=tracking_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
