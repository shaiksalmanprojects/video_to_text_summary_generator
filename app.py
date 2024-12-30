import os
import streamlit as st
import yt_dlp
import ffmpeg
import whisper
from transformers import pipeline

# Directory Setup
DOWNLOAD_DIR = "downloads"
AUDIO_DIR = "audio"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Helper Functions
def download_video(url):
    """Download video from YouTube."""
    ydl_opts = {
        'format': 'best',
        'outtmpl': f'{DOWNLOAD_DIR}/%(title)s.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = os.path.join(DOWNLOAD_DIR, f"{info['title']}.{info['ext']}")
        return video_path

def extract_audio(video_path):
    """Extract audio from video."""
    audio_path = os.path.join(AUDIO_DIR, "audio.mp3")
    ffmpeg.input(video_path).output(audio_path, format="mp3", acodec="mp3").run(overwrite_output=True)
    return audio_path

def transcribe_audio(audio_path, model_name="base"):
    """Transcribe audio using Whisper."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"]

# Streamlit App
st.title("Video Description Generator")
st.write("Upload a YouTube video link to generate a description using Whisper and Hugging Face.")

# Input YouTube URL
url = st.text_input("Enter YouTube URL:")

if st.button("Process Video") and url:
    st.write("Downloading video...")
    video_path = download_video(url)
    st.write(f"Video downloaded to: {video_path}")
    audio_path = extract_audio(video_path)
    transcription = transcribe_audio(audio_path)
    st.write("Description :")
    st.write(transcription)
