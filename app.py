import cv2
import streamlit as st
from ultralytics import YOLO
import os

def app():
    st.header('Object Detection Web App')
    st.subheader('Powered by YOLOv8')
    st.write('Howdy!')
    model = YOLO('yolov8n.pt') # Capable of up to 80 classes
    object_names = list(model.names.values())

    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload video", type=['mp4'])
        selected_objects = st.multiselect('Choose objects to detect', object_names, default=['person']) # Specify which class to visualize
        min_confidence = st.slider('Confidence score', 0.0, 1.0)
        st.form_submit_button(label='Submit')
    
    if uploaded_file is not None: 
        input_path = uploaded_file.name
        file_binary = uploaded_file.read()
        with open(input_path, "wb") as temp_file:
            temp_file.write(file_binary)
        video_stream = cv2.VideoCapture(input_path) # Load the uploaded video
        width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'h264') # Video compression standard passed to create code for H.264 Codec.
        fps = int(video_stream.get(cv2.CAP_PROP_FPS))
        output_path = input_path.split('.')[0] + '_output.mp4'
        out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height))

        # Spinning Widget telling user that video is being processed
        with st.spinner('Processing Video...'):
            while True:
                ret, frame = video_stream.read() # Inside a loop, read a frame from video, stop when there is any error
                if not ret:
                    break
                result = model(frame) # prediction on the frame
                # Get the bounding box coordinates, score, and class name for every detected objects
                for detection in result[0].boxes.data:
                    x0, y0 = (int(detection[0]), int(detection[1]))
                    x1, y1 = (int(detection[2]), int(detection[3]))
                    score = round(float(detection[4]), 2)
                    cls = int(detection[5])
                    object_name = model.names[cls]
                    label = f'{object_name} {score}'
                    # Draw bounding boxes and labels if object in selected categories, and score > confidence threshold
                    if model.names[cls] in selected_objects and score > min_confidence:
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Display number of detected objects on the processed frame
                detections = result[0].verbose() # Retrieve string that tells class and number of objecdts detected
                cv2.putText(frame, detections, (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Save the processed video
                out_video.write(frame)

            video_stream.release()
            out_video.release()
        
        # Display the processed video in Streamlit app, if the file exists
        if not os.path.exists(output_path):
            st.error("Failed to process video")
        else:
            st.video(output_path)

if __name__ == "__main__":
    app()