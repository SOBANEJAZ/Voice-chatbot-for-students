import gradio as gr
import pyaudio, wave, keyboard, torch, os
from openai import OpenAI
from faster_whisper import WhisperModel
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

system_prompt = {
    'role': 'system', 
    'content': 'You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk is carefree, sometimes a bit sarky, but always cool. In our story, you\'re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremely short and crisp, only very few words, as if we\'re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia.'
}

model = WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu')
history = []

def generate(messages):
    answer = ""
    for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            answer += text_chunk
            yield text_chunk
    return answer

def transcribe_recording():
    audio = pyaudio.PyAudio()
    frames = []
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    
    def callback(in_data, frame_count, time_info, status):
        frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    
    py_stream.start_stream()
    while True:
        if keyboard.is_pressed('space'):
            py_stream.stop_stream()
            break
    
    py_stream.close()
    audio.terminate()
    
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    
    user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="en")[0])
    history.append({'role': 'user', 'content': user_text})
    
    return user_text

def chat_with_sophia():
    generator = generate([system_prompt] + history[-10:])
    answer = "".join(list(generator))
    stream(elevenlabs_client.generate(text=answer, voice="Nicole", model="eleven_monolingual_v1", stream=True))
    history.append({'role': 'assistant', 'content': answer})
    return answer

def start_recording():
    return "Recording started. Press space to stop."

def stop_recording():
    user_text = transcribe_recording()
    response = chat_with_sophia()
    return user_text, response

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Sophia Chatbot with Voice Interface")
    with gr.Row():
        with gr.Column():
            start_button = gr.Button("Start Recording")
            stop_button = gr.Button("Stop Recording")
        with gr.Column():
            transcribed_text = gr.Textbox(label="Transcribed Text")
            chatbot_response = gr.Textbox(label="Chatbot Response")
    
    start_button.click(fn=start_recording, outputs=[])
    stop_button.click(fn=stop_recording, outputs=[transcribed_text, chatbot_response])

demo.launch()
