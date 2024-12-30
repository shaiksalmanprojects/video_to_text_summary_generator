import os
import streamlit as st
import cv2
import numpy as np
from transformers import pipeline
import torch

# Directories Setup
FRAME_DIR = "frames"
os.makedirs(FRAME_DIR, exist_ok=True)

# Load Pre-trained Vision Model (e.g., CLIP or ViT)
vision_model = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Helper Functions

def extract_frames(video_path, interval=30):
    """Extract frames from a video at regular intervals."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            frame_path = os.path.join(FRAME_DIR, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
            extracted_frames.append(frame_path)
        
        frame_count += 1
    
    cap.release()
    return extracted_frames

def generate_descriptions(frames):
    """Generate text descriptions from video frames."""
    descriptions = []
    for frame in frames:
        result = vision_model(frame)
        descriptions.append(result[0]['generated_text'])
    return descriptions

def summarize_descriptions(descriptions):
    """Summarize frame descriptions into a cohesive narrative."""
    summarizer = pipeline("summarization")
    combined_text = " ".join(descriptions)
    summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit App
st.title("Video-to-Text Description AI Model")
st.write("Upload a video file or provide a YouTube URL to generate a textual description.")

# Video Upload
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    video_path = os.path.join("uploads", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())
    
    st.write("Extracting frames...")
    frames = extract_frames(video_path)
    
    st.write("Generating descriptions...")
    descriptions = generate_descriptions(frames)
    st.write(descriptions)
    
    st.write("Summarizing descriptions...")
    summary = summarize_descriptions(descriptions)
    
    st.write("### Video Description:")
    st.write(summary)

st.write("Note: This model generates text directly from visual content without using audio or OCR.")
