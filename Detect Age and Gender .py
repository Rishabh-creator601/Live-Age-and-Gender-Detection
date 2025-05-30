import streamlit as st
import av, cv2, time, os 
import numpy as np
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from model_usage import predict_age_gender
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 



detector = cv2.CascadeClassifier(os.path.join("./models/haarcascade_frontalface_default.xml"))


st.title("Live Age and Gender  Detection with WebRTC")
st.sidebar.success("Select a page above.")



class Age_GenderDetection(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.last_time = time.time()

    def recv(self, frame):
        self.frame_count += 1
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        # FPS calculation
        now = time.time()
        fps = 1 / (now - self.last_time)
        self.last_time = now

        # Skip frames for performance
        

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi = roi_gray.astype("float") / 255.0
           
            roi = img_to_array(roi)
            age, gender =  predict_age_gender(roi)
            
            label = f"{age} | {gender}"
            

            cv2.putText(image, label, (x, y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # if self.frame_count % 5 != 0:
        return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_streamer(
    key="age-gender-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=Age_GenderDetection,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)