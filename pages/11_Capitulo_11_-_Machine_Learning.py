import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np
import av

st.set_page_config(page_title="Capítulo 11: Machine Learning by an Artificial Neural Network", page_icon="🔎")
st.title('Capítulo 11: Machine Learning by an Artificial Neural Network')

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text("Letra A")
        st.image('./guias/letra_a.png', caption='Figura 11.1: Portada del capítulo 11.')
    with col2:
        st.text("Letra B")
        st.image('./guias/letra_b.png', caption='Figura 11.2: Portada del capítulo 11.')
    with col3:
        st.text("Letra C")
        st.image('./guias/letra_c.png', caption='Figura 11.3: Portada del capítulo 11.')

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.text("Letra D")
        st.image('./guias/letra_d.png', caption='Figura 11.4: Portada del capítulo 11.')
    with col2:
        st.text("Letra E")
        st.image('./guias/letra_e.png', caption='Figura 11.5: Portada del capítulo 11.')
    with col3:
        st.text("Letra F")
        st.image('./guias/letra_f.png', caption='Figura 11.6: Portada del capítulo 11.')

# Todo el crédito es de https://github.com/SebasProgrammer/abecedario_A-F#.
# Yo solo agregue la compatibilidad con streamlit y la webcam.

def draw_text_with_border(image, text, position, font=cv2.FONT_HERSHEY_DUPLEX, 
                          font_scale=5, color=(0, 0, 255), thickness=10, border_color=(0, 0, 0)):
    """
    Dibuja texto con borde negro para mejor visibilidad
    """
    x, y = position
    # Dibujar borde negro
    cv2.putText(image, text, (x-2, y-2), font, font_scale, border_color, thickness+2, cv2.LINE_AA)
    cv2.putText(image, text, (x+2, y-2), font, font_scale, border_color, thickness+2, cv2.LINE_AA)
    cv2.putText(image, text, (x-2, y+2), font, font_scale, border_color, thickness+2, cv2.LINE_AA)
    cv2.putText(image, text, (x+2, y+2), font, font_scale, border_color, thickness+2, cv2.LINE_AA)
    # Dibujar texto principal
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    # Iterate through the landmarks to find the bounding box coordinates
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def evaluarCap(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1) as hands:

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_height, image_width, _ = image.shape
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks):
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Draw bounding box
                    draw_bounding_box(image, hand_landmarks)

                    index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                    int(hand_landmarks.landmark[8].y * image_height))
                    index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                    int(hand_landmarks.landmark[6].y * image_height))
                    
                    thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                                    int(hand_landmarks.landmark[4].y * image_height))
                    thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                                    int(hand_landmarks.landmark[2].y * image_height))
                    
                    middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                    int(hand_landmarks.landmark[12].y * image_height))
                    
                    middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                                    int(hand_landmarks.landmark[10].y * image_height))
                    
                    ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                                    int(hand_landmarks.landmark[16].y * image_height))
                    ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                                    int(hand_landmarks.landmark[14].y * image_height))
                    
                    pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                                    int(hand_landmarks.landmark[20].y * image_height))
                    pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                                    int(hand_landmarks.landmark[18].y * image_height))
                    
                    wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                    int(hand_landmarks.landmark[0].y * image_height))
                    
                    ring_finger_pip2 = (int(hand_landmarks.landmark[5].x * image_width),
                                    int(hand_landmarks.landmark[5].y * image_height))
                    
                    if abs(thumb_tip[1] - index_finger_pip[1]) <45 \
                        and abs(thumb_tip[1] - middle_finger_pip[1]) < 30 and abs(thumb_tip[1] - ring_finger_pip[1]) < 30\
                        and abs(thumb_tip[1] - pinky_pip[1]) < 30:
                        image = draw_text_with_border(image, 'A', (image_width-150, 120))
                        
                    
                    elif index_finger_pip[1] - index_finger_tip[1]>0 and pinky_pip[1] - pinky_tip[1] > 0 and \
                        middle_finger_pip[1] - middle_finger_tip[1] >0 and ring_finger_pip[1] - ring_finger_tip[1] >0 and \
                            middle_finger_tip[1] - ring_finger_tip[1] <0 and abs(thumb_tip[1] - ring_finger_pip2[1])<40:
                        image = draw_text_with_border(image, 'B', (image_width-150, 120))
                        
                    elif abs(index_finger_tip[1] - thumb_tip[1]) < 360 and \
                        index_finger_tip[1] - middle_finger_pip[1]<0 and index_finger_tip[1] - middle_finger_tip[1] < 0 and \
                            index_finger_tip[1] - index_finger_pip[1] > 0:
                        image = draw_text_with_border(image, 'C', (image_width-150, 120))
                    
                    elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 \
                        and distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 \
                        and  pinky_pip[1] - pinky_tip[1]<0\
                        and index_finger_pip[1] - index_finger_tip[1]>0:
                        image = draw_text_with_border(image, 'D', (image_width-150, 120))
                    
                    elif index_finger_pip[1] - index_finger_tip[1] < 0 and pinky_pip[1] - pinky_tip[1] < 0 and \
                        middle_finger_pip[1] - middle_finger_tip[1] < 0 and ring_finger_pip[1] - ring_finger_tip[1] < 0 \
                            and abs(index_finger_tip[1] - thumb_tip[1]) < 100 and \
                                thumb_tip[1] - index_finger_tip[1] > 0 \
                                and thumb_tip[1] - middle_finger_tip[1] > 0 \
                                and thumb_tip[1] - ring_finger_tip[1] > 0 \
                                and thumb_tip[1] - pinky_tip[1] > 0:

                        image = draw_text_with_border(image, 'E', (image_width-150, 120))
                        
                    elif  pinky_pip[1] - pinky_tip[1] > 0 and middle_finger_pip[1] - middle_finger_tip[1] > 0 and \
                        ring_finger_pip[1] - ring_finger_tip[1] > 0 and index_finger_pip[1] - index_finger_tip[1] < 0 \
                            and abs(thumb_pip[1] - thumb_tip[1]) > 0 and distancia_euclidiana(index_finger_tip, thumb_tip) <65:

                        image = draw_text_with_border(image, 'F', (image_width-150, 120))

        return av.VideoFrame.from_ndarray(image, format="bgr24")
video = webrtc_streamer(
    key="sistinteligentes11",
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    video_frame_callback=evaluarCap,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)