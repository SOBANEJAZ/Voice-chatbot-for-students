import streamlit as st
from openai import OpenAI
import pyaudio, wave, faster_whisper, torch.cuda, os
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
import tempfile
import time
import threading

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
system_prompt = {
    'role': 'system', 
    'content': 'You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk is carefree, sometimes a bit sarky, but always cool. In our story, you\'re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremely short and crisp, only very few words, as if we\'re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia.'
}

@st.cache_resource
def get_whisper_model():
    return faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu')

def generate(messages):
    answer = ""
    for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            answer += text_chunk
            yield text_chunk
    return answer

def record_audio():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []
    
    start_time = time.time()
    while time.time() - start_time < 10:  # Record for 10 seconds
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    return frames

st.title("Sophia Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

if "recording" not in st.session_state:
    st.session_state.recording = False

if "audio_frames" not in st.session_state:
    st.session_state.audio_frames = None

# Audio recording
if st.button("Record (10 seconds)"):
    st.session_state.recording = True
    st.session_state.audio_frames = record_audio()
    st.session_state.recording = False
    st.rerun()

if st.session_state.audio_frames:
    # Save the recorded audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        wf = wave.open(tmpfile.name, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(2)  # Assuming 16-bit audio
        wf.setframerate(16000)
        wf.writeframes(b''.join(st.session_state.audio_frames))
        wf.close()
        
        # Transcribe the audio
        model = get_whisper_model()
        user_text = " ".join(seg.text for seg in model.transcribe(tmpfile.name, language="en")[0])
        
    st.write(f"You said: {user_text}")
    st.session_state.history.append({'role': 'user', 'content': user_text})

    # Generate response
    messages = [system_prompt] + st.session_state.history[-10:]
    response_container = st.empty()
    full_response = ""
    
    for chunk in generate(messages):
        full_response += chunk
        response_container.markdown(full_response)
    
    st.session_state.history.append({'role': 'assistant', 'content': full_response})
    
    # Stream the audio response
    audio_stream = elevenlabs_client.generate(text=full_response, voice="Nicole", model="eleven_monolingual_v1", stream=True)
    audio_data = b"".join(chunk for chunk in audio_stream)
    st.audio(audio_data, format="audio/mp3")
    
    # Clear the audio frames after processing
    st.session_state.audio_frames = None

# Display conversation history
for message in st.session_state.history:
    if message['role'] == 'user':
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Sophia: {message['content']}")