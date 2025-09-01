import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
from keras.models import load_model
import numpy as np
import webbrowser
import altair
import av  # Import av for video processing
import logging


logging.basicConfig(level=logging.DEBUG)

# Check Altair Version
print(altair.__version__)

# Load Pre-trained Emotion Detection Model
model_path = "C:\\Users\\nitin yadav\\Downloads\\Music-Recommendation-Using-Facial-Expressions-main\\Music-Recommendation-Using-Facial-Expressions-main\\code\\model\\fer2013_mini_XCEPTION.102-0.66.hdf5"
try:
    model = load_model(model_path, compile=False)  # Avoid compiling during load
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Load OpenCV Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Emotion Labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize Emotion State
if "emotion" not in st.session_state:
    st.session_state["emotion"] = "Neutral"


class EmotionProcessor(VideoProcessorBase):
    """Processes video frames, detects faces, and predicts emotions."""
    
    def __init__(self):
        self.emotion_label = "Neutral"

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Detect faces and predict emotion
        if len(faces) > 0:  # If face is detected
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
                roi = roi_gray / 255.0
                roi = np.reshape(roi, (1, 64, 64, 1))

                try:
                    prediction = model.predict(roi)
                    self.emotion_label = emotions[np.argmax(prediction)]
                except Exception as e:
                    self.emotion_label = "Error"
                    st.error(f"Model prediction error: {e}")

                # Draw emotion text on frame
                cv2.rectangle(frm, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frm, self.emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Save detected emotion in session state
            st.session_state["emotion"] = self.emotion_label
        else:
            self.emotion_label = "No face detected"
            st.session_state["emotion"] = self.emotion_label

        return av.VideoFrame.from_ndarray(frm, format="bgr24")  # Fix frame return


def recommend_song():
    """Recommends a song based on detected emotion using YouTube."""
    emotion = st.session_state["emotion"]
    
    if emotion == "Neutral" or emotion == "No face detected":
        st.warning("No strong emotion detected. Try again!")
        return

    search_query = f"https://www.youtube.com/results?search_query={emotion}+mood+music"
    webbrowser.open(search_query)
    st.success(f"🎵 Opening YouTube songs for: {emotion}")


# Streamlit UI
st.title("🎭 Emotion-Based Music Recommender 🎶")
st.write("This app detects your facial expression and recommends a song based on your mood!")

# Start WebRTC Video Stream
webrtc_streamer(
    key="emotion_recognition",
    video_processor_factory=EmotionProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"facingMode": "user"}, "audio": False}
)



# Button to recommend a song
if st.button("🎵 Recommend a Song 🎶"):
    recommend_song()
