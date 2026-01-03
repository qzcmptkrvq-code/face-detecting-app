import streamlit as st
import cv2

st.title("🟢 Live Face Detection Web App")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Placeholder for live feed
frame_placeholder = st.empty()

# Button to start camera
if st.button("Camera"):
    cap = cv2.VideoCapture(0)
    st.info("Camera started. Press 'Stop' button to exit.")

    # Stop button
    stop_button = st.button("Stop")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from camera.")
            break

        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw green rectangles on original BGR frame
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Convert BGR → RGB before displaying in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        # Break if stop button pressed
        if stop_button:
            break

    cap.release()
    st.success("Camera stopped.")

