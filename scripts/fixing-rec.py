import streamlit as st
import pyaudio
import wave
import os
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import faster_whisper
import torch

from dotenv import load_dotenv, find_dotenv

# Load environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Set up API clients
openai_api = os.getenv("openai")
elevenlabs_api = os.getenv("elevenlabs")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
openai_client = OpenAI(api_key=openai_api)
elevenlabs_client = ElevenLabs(api_key=elevenlabs_api)

# Initialize Whisper model
model = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu')

# Define the system prompt
system_prompt = {
    'role': 'system',
    'content': "You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk is carefree, sometimes a bit sarky, but always cool. In our story, you're not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremely short and crisp, only very few words, as if we're in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia."
}

# Initialize chat history
history = []

# Streamlit UI
st.title("Voice Chatbot with Streamlit")

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Function to record audio
def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    frames = []

    st.info("Recording... Press 'Stop' to end recording.")
    while not st.button('Stop'):
        frames.append(stream.read(512))

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    
    return "voice_record.wav"

# Function to transcribe audio
def transcribe_audio(file_path):
    segments, _ = model.transcribe(file_path, language="en")
    return " ".join([seg.text for seg in segments])

# Function to generate chatbot response
def generate_response(messages):
    response = ""
    for chunk in openai_client.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            response += text_chunk
            yield text_chunk

# Record button
if st.button('Record'):
    audio_file = record_audio()
    user_text = transcribe_audio(audio_file)
    st.write(f"**User:** {user_text}")
    st.session_state['history'].append({'role': 'user', 'content': user_text})
    
    generator = generate_response([system_prompt] + st.session_state['history'][-10:])
    response_text = ''.join(list(generator))
    st.write(f"**Sophia:** {response_text}")
    st.session_state['history'].append({'role': 'assistant', 'content': response_text})
    
    # Stream the response with ElevenLabs
    stream(elevenlabs_client.generate(text=response_text, voice="Nicole", model="eleven_monolingual_v1", stream=True))

# Display chat history
if st.session_state['history']:
    for message in st.session_state['history']:
        if message['role'] == 'user':
            st.write(f"**User:** {message['content']}")
        else:
            st.write(f"**Sophia:** {message['content']}")
