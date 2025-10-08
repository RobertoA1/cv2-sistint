import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np

st.set_page_config(page_title="Capítulo 10: Sombrero con Realidad Aumentada", page_icon="🎩")
st.title('Capítulo 10: Sombrero con Realidad Aumentada')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def create_hat_image():
    hat = np.zeros((100, 200, 4), dtype=np.uint8)
    cv2.rectangle(hat, (50, 30), (150, 60), (50, 50, 50, 255), -1)
    cv2.ellipse(hat, (100, 60), (90, 20), 0, 0, 360, (50, 50, 50, 255), -1)
    cv2.rectangle(hat, (50, 55), (150, 65), (200, 50, 50, 255), -1)
    return hat

hat_image = create_hat_image()

def overlay_image(background, overlay, x, y, angle=0):
    if angle != 0:
        h, w = overlay.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        overlay = cv2.warpAffine(overlay, M, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    
    h, w = overlay.shape[:2]
    x1 = x - w // 2
    y1 = y - h // 2
    x2 = x1 + w
    y2 = y1 + h
    
    if x1 >= background.shape[1] or y1 >= background.shape[0] or x2 <= 0 or y2 <= 0:
        return background
    
    overlay_x1 = max(0, -x1)
    overlay_y1 = max(0, -y1)
    overlay_x2 = w - max(0, x2 - background.shape[1])
    overlay_y2 = h - max(0, y2 - background.shape[0])
    
    bg_x1 = max(0, x1)
    bg_y1 = max(0, y1)
    bg_x2 = min(background.shape[1], x2)
    bg_y2 = min(background.shape[0], y2)
    
    if overlay_x2 <= overlay_x1 or overlay_y2 <= overlay_y1:
        return background
    
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    background_crop = background[bg_y1:bg_y2, bg_x1:bg_x2]
    overlay_rgb = overlay_crop[:, :, :3]
    overlay_alpha = overlay_crop[:, :, 3:] / 255.0
    blended = overlay_rgb * overlay_alpha + background_crop * (1 - overlay_alpha)
    background[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)
    
    return background

def estimate_face_angle(face_region):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.05, minNeighbors=2, minSize=(10, 10))
    
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda x: x[0])
        eye1 = eyes[0]
        eye2 = eyes[1]
        eye1_center = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
        eye2_center = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)
        dy = eye2_center[1] - eye1_center[1]
        dx = eye2_center[0] - eye1_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        return -angle
    
    return 0

def ar_hat_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        angle = estimate_face_angle(face_gray)
        hat_width = int(w * 1.5)
        hat_height = int(hat_width * 0.5)
        hat_resized = cv2.resize(hat_image, (hat_width, hat_height))
        hat_x = x + w // 2
        hat_y = y - int(hat_height * 0.6)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        img = overlay_image(img, hat_resized, hat_x, hat_y, angle)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="sistinteligentes11",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_frame_callback=ar_hat_frame,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
