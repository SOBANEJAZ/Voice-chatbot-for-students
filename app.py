# Set up the Whisper model
import streamlit as st
import pyaudio
import wave
import numpy as np
import collections
import faster_whisper
import torch.cuda
import os
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
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

# Set up the Whisper model
model = faster_whisper.WhisperModel(
    model_size_or_path="tiny", 
    device='cuda' if torch.cuda.is_available() else 'cpu', compute_type='float32'
)

# System prompt
system_prompt = {
    'role': 'system',
    'content': (
        'You are Sophia. You are an exciting girl who loves to chat with people.'
    )
}

# Initialize history
history = []

def generate(messages):
    answer = ""        
    for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            answer += text_chunk
            yield text_chunk
    return answer

def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level

def record_audio():
    audio = pyaudio.PyAudio()
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
    frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False

    st.write("Start speaking.")
    while True:
        data = py_stream.read(512)
        pegel, long_term_noise_level, current_noise_level = get_levels(data, long_term_noise_level, current_noise_level)
        audio_buffer.append(data)

        if voice_activity_detected:
            frames.append(data)
            if current_noise_level < ambient_noise_level + 100:
                break  # voice activity ends

        if not voice_activity_detected and current_noise_level > long_term_noise_level + 300:
            voice_activity_detected = True
            st.write("I'm all ears.")
            ambient_noise_level = long_term_noise_level
            frames.extend(list(audio_buffer))

    py_stream.stop_stream()
    py_stream.close()
    audio.terminate()

    # Save recording
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))

    return "voice_record.wav"

# Streamlit app
st.title("Sophia Chatbot")
st.write("Press the button and start speaking to interact with Sophia.")

if st.button("Start Recording"):
    audio_file = record_audio()
    user_text = " ".join(seg.text for seg in model.transcribe(audio_file, language="en")[0])
    st.write(f'You said: {user_text}')
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate([system_prompt] + history[-10:])
    generated_text = "".join(list(generator))
    history.append({'role': 'assistant', 'content': generated_text})
    st.write(f'Sophia says: {generated_text}')

    # Stream the response using ElevenLabs
    stream(elevenlabs_client.generate(text=generated_text, voice="Nicole", model="eleven_multilingual_v2", stream=True))
