from openai import OpenAI
import pyaudio, wave, keyboard, faster_whisper, torch.cuda, os
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

model, answer, history = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu'), "", []

def generate(messages):
    global answer
    answer = ""
        
    for chunk in openai_client.chat.completions.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk.choices[0].delta.content):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk


while True:
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    # Record from microphone until user presses space bar again
    print("I'm all ears. Tap space when you're done.\n")
    audio, frames = pyaudio.PyAudio(), []
    py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'): 
        frames.append(py_stream.read(512))
    py_stream.stop_stream(), py_stream.close(), audio.terminate()

    # Transcribe recording using whisper
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="en")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate([system_prompt] + history[-10:])
    stream(elevenlabs_client.generate(text=generator, voice="Nicole", model="eleven_monolingual_v1", stream=True))
    history.append({'role': 'assistant', 'content': answer})