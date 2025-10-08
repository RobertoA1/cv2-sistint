from streamlit_webrtc import webrtc_streamer
import cv2
import av
import numpy as np
import streamlit as st

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')

if "lentes_path" not in st.session_state:
    st.session_state['lentes_path'] = './images/lentes1.png'
if "factor_tamaño_lentes" not in st.session_state:
    st.session_state["factor_tamaño_lentes"] = 0.85
if "factor_altura_lentes" not in st.session_state:
    st.session_state["factor_altura_lentes"] = 0.7

sunglasses_img = cv2.imread(st.session_state["lentes_path"], cv2.IMREAD_UNCHANGED)

factor_tamaño_lentes = st.session_state['factor_tamaño_lentes']
factor_altura_lentes = st.session_state['factor_altura_lentes']

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def eliminarFondo(frame: av.VideoFrame) -> av.VideoFrame:
    cap = frame.to_ndarray(format="bgr24")
    
    frame = unsharp_mask(cap)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    centers = []

    for (x,y,w,h) in faces: 
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = frame[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray) 
        for (x_eye,y_eye,w_eye,h_eye) in eyes: 
            centers.append((x + int(x_eye + 0.5*w_eye), y + int(y_eye + 0.5*h_eye))) 
    
    if len(centers) > 1: # if detects both eyes
        h, w = sunglasses_img.shape[:2]
        eye_distance = abs(centers[1][0] - centers[0][0])
        sunglasses_width = 2.12 * eye_distance * (1/(1.8 - factor_tamaño_lentes))
        scaling_factor = sunglasses_width / w
        overlay_sunglasses = cv2.resize(sunglasses_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
        x -= int(0.26*overlay_sunglasses.shape[1])
        y += int(factor_altura_lentes*overlay_sunglasses.shape[0])

        h, w = overlay_sunglasses.shape[:2]
        h, w = int(h), int(w)
        frame_roi = frame[y:y+h, x:x+w]

        # Separar canales BGR y alfa
        if overlay_sunglasses.shape[2] == 4:
            bgr_sunglasses = overlay_sunglasses[:, :, :3]
            alpha_sunglasses = overlay_sunglasses[:, :, 3]
            # Crear máscara y su inversa
            mask = cv2.threshold(alpha_sunglasses, 180, 255, cv2.THRESH_BINARY)[1]
            mask_inv = cv2.bitwise_not(mask)
        else:
            bgr_sunglasses = overlay_sunglasses
            gray_overlay_sunglassess = cv2.cvtColor(bgr_sunglasses, cv2.COLOR_BGR2GRAY)
            mask = cv2.threshold(gray_overlay_sunglassess, 180, 255, cv2.THRESH_BINARY_INV)[1]
            mask_inv = cv2.bitwise_not(mask)

        try:
            masked_face = cv2.bitwise_and(bgr_sunglasses, bgr_sunglasses, mask=mask)
            masked_frame = cv2.bitwise_and(frame_roi, frame_roi, mask=mask_inv)
        except cv2.error as e:
            print('Ignoring arithmetic exceptions: '+ str(e))

        frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")

css = """
    <style>
        .stVerticalBlock {
            display: grid !important;
            grid-template-rows: 200px 200px 1fr 100px !important;
            grid-template-columns: 1fr 1fr 1fr !important;
        }

        .stVerticalBlock > * {
            grid-column: 1 / 4 !important;    
        }

        .stButton {
            display: inline-block !important;
            width: 200px !important;
            height: 200px !important;
        }

        .stButton button {
            width: 100% !important;
            height: 100% !important;
        }

        .stButton button div {
            width: inherit !important;
            height: inherit !important;
        }

        .stButton button div p {
            display: block !important;
            width: inherit !important;
            height: inherit !important;
        }

        .stButton button div p img {
            height: 100% !important;
            max-height: 100% !important;
        }

        .st-key-selector-lentes {
            display: flex !important;
            justify-content: center !important;
            gap: 20px !important;
            flex-direction: row !important;
        }
    </style>
"""

st.set_page_config(
    page_title="Capítulo 04: Fun with eyes",
    page_icon="😎"
)

st.title('Capítulo 04: Fun with eyes')

with st.container(key='selector-lentes'):
    botonLentes1 = st.button("![Lentes 1](https://i.imgur.com/2mTPw0l_d.webp)")
    botonLentes2 = st.button("![Lentes 2](https://i.imgur.com/7nLFjDE.png)")
    botonLentes3 = st.button("![Lentes 3](https://i.imgur.com/YNxYJJw.png)")

st.html(css)

if botonLentes1:
    st.session_state["lentes_path"] = './images/lentes1.png'
    sunglasses_img = cv2.imread('./images/lentes1.png', cv2.IMREAD_UNCHANGED)
elif botonLentes2:
    st.session_state["lentes_path"] = './images/lentes2.png'
    sunglasses_img = cv2.imread('./images/lentes2.png', cv2.IMREAD_UNCHANGED)
elif botonLentes3:
    st.session_state["lentes_path"] = './images/lentes3.png'
    sunglasses_img = cv2.imread('./images/lentes3.png', cv2.IMREAD_UNCHANGED)

video = webrtc_streamer(
    key="sistinteligentes4",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_frame_callback=eliminarFondo,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

with st.container(key='selector-parametros'):
    col1, col2 = st.columns(2)
    with col1:
        tamaño = st.slider('Tamaño de los Lentes', 0.8, 1.0, 0.85)
        factor_tamaño_lentes = tamaño
    with col2:
        alturaOjos = st.slider('Posición Y de los lentes', 0.0, 1.0, 0.7)
        factor_altura_lentes = alturaOjos

st.session_state["factor_tamaño_lentes"] = tamaño
st.session_state["factor_altura_lentes"] = alturaOjos