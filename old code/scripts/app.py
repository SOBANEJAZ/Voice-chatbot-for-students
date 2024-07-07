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
    device='cuda' if torch.cuda.is_available() else 'cpu', compute_type="float32"
)

# System prompt
system_prompt = {
    'role': 'system',
    'content': (
        "You are a Student Assistant. Your task is to engage students in conversation based on their course day. Follow these steps:\n\n"
        "1. Ask the student for the day of the course.\n"
        "2. After the student provides the day, give them a topic from that day.\n"
        "3. Ask a related question about the topic.\n"
        "4. Follow up on their answer with three or four more related questions.\n"
        "5. If the student asks to change the question or after three related questions, choose another question from the same day.\n\n"
        "Additionally, follow these guidelines for clarity and assistance:\n"
        "- Rephrase and repeat the student’s answers to confirm understanding: 'Oh, you mean...'\n"
        "- If the student asks for help, you can:\n"
        "  - Clarify or correct mistakes.\n"
        "  - Provide examples.\n"
        "  - Explain in a different way.\n\n"
        "Course Topics:\n\n"
        "Week 1\n"
        "Day 1: Introduce Yourself\n"
        "1. What are your aspirations for the future?\n"
        "2. Describe a significant achievement in your life.\n"
        "3. How do you handle challenges in your life?\n"
        "4. Share a memorable childhood experience.\n"
        "5. What motivates you on a daily basis?\n"
        "6. Describe a hobby you're passionate about.\n"
        "7. How do you spend your weekends?\n"
        "8. What's your favorite genre of music?\n"
        "9. How do you like to relax after a long day?\n"
        "10. Share a funny anecdote about yourself.\n"
        "11. What role does art play in your life?\n"
        "12. Describe your dream vacation.\n"
        "13. How do you handle stress?\n"
        "14. Share a goal you're currently working towards.\n"
        "15. What's the most adventurous thing you've done?\n"
        "16. How do you like to spend your holidays?\n"
        "17. Describe a favorite childhood memory.\n"
        "18. What cultural traditions are important to you?\n"
        "19. Share a book that has influenced you.\n"
        "20. How do you balance work and personal life?\n\n"
        "Day 2: Daily Routine\n"
        "1. How do you stay productive throughout the day?\n"
        "2. Describe your morning routine.\n"
        "3. What's your favorite breakfast dish?\n"
        "4. How do you unwind after work or school?\n"
        "5. Share a daily habit that improves your life.\n"
        "6. Describe your evening routine.\n"
        "7. How do you prepare for the week ahead?\n"
        "8. What's your favorite way to relax at home?\n"
        "9. How do you prioritize tasks during the day?\n"
        "10. Share a strategy for staying organized.\n"
        "11. What time do you usually go to bed?\n"
        "12. Describe a typical workday or school day.\n"
        "13. How do you incorporate exercise into your routine?\n"
        "14. Share a tip for starting the day positively.\n"
        "15. What's your favorite part of the day?\n"
        "16. How do you handle unexpected changes to your routine?\n"
        "17. Describe a morning ritual you enjoy.\n"
        "18. Share a favorite quote that inspires you.\n"
        "19. How do you reflect on your day?\n"
        "20. What's your go-to stress-relief technique?\n\n"
        "Week 2\n"
        "Day 8: Weather\n"
        "1. How does weather affect your mood?\n"
        "2. Describe your favorite weather to experience.\n"
        "3. What outdoor activities do you enjoy in different seasons?\n"
        "4. How do you dress for different weather conditions?\n"
        "5. Share a memorable weather event you experienced.\n"
        "6. How do you prepare for extreme weather?\n"
        "7. What's your least favorite weather and why?\n"
        "8. Describe a place with the best weather you've visited.\n"
        "9. How do you enjoy rainy days?\n"
        "10. Share a story about a weather-related adventure.\n"
        "11. What's your favorite season for outdoor activities?\n"
        "12. How do you stay safe during storms?\n"
        "13. Describe a weather phenomenon you find fascinating.\n"
        "14. What's your favorite piece of clothing for cold weather?\n"
        "15. How do you celebrate sunny days?\n"
        "16. Share a weather-related tradition you have.\n"
        "17. How do you check the weather forecast?\n"
        "18. Describe your ideal weather for a vacation.\n"
        "19. Share a tip for staying comfortable in hot weather.\n"
        "20. How do you adapt your plans based on weather forecasts?\n\n"
        "Day 9: Shopping\n"
        "1. What's your approach to shopping for clothes?\n"
        "2. Describe a recent shopping bargain you found.\n"
        "3. How do you decide what to buy?\n"
        "4. Share a story about a memorable shopping trip.\n"
        "5. What's your favorite store to browse?\n"
        "6. How do you feel about shopping online versus in-store?\n"
        "7. Describe your shopping strategy during sales.\n"
        "8. Share a tip for finding the best deals.\n"
        "9. How do you budget for shopping trips?\n"
        "10. What's your guilty pleasure when shopping?\n"
        "11. Share a shopping regret and what you learned from it.\n"
        "12. How do you organize your shopping list?\n"
        "13. Describe a favorite shopping memory.\n"
        "14. What's your go-to gift for friends or family?\n"
        "15. How do you avoid impulse purchases?\n"
        "16. Share a story about a unique item you bought.\n"
        "17. How do you find sustainable fashion options?\n"
        "18. Describe your shopping routine on weekends.\n"
        "19. What's your favorite section in a department store?\n"
        "20. How do you shop for groceries efficiently?\n\n"
        "Week 3\n"
        "Day 15: Best Friend\n"
        "1. What qualities do you value most in a friend?\n"
        "2. Describe your best friend's personality.\n"
        "3. How do you support each other during tough times?\n"
        "4. Share a funny memory with your best friend.\n"
        "5. How did you meet your best friend?\n"
        "6. Describe your favorite activity to do together.\n"
        "7. How often do you communicate with your best friend?\n"
        "8. Share a story about a meaningful gift from your best friend.\n"
        "9. What's your favorite inside joke with your best friend?\n"
        "10. How do you celebrate each other's achievements?\n"
        "11. Describe a time your best friend surprised you.\n"
        "12. How do you resolve conflicts with your best friend?\n"
        "13. Share a favorite adventure you've had together.\n"
        "14. What advice would you give to new friends?\n"
        "15. How do you stay connected over long distances?\n"
        "16. Describe a trip you've taken with your best friend.\n"
        "17. What's your favorite memory together?\n"
        "18. How do you celebrate special occasions with your best friend?\n"
        "19. Share a lesson you've learned from your best friend.\n"
        "20. How do you support each other's personal growth?\n\n"
        "Day 16: Cooking\n"
        "1. How did you learn to cook?\n"
        "2. Describe a cooking disaster and what you learned.\n"
        "3. What's your signature dish to cook for guests?\n"
        "4. How do you plan meals for the week?\n"
        "5. Share a cooking tip that changed your life.\n"
        "6. What's your favorite cuisine to cook?\n"
        "7. Describe a memorable meal you've cooked.\n"
        "8. How do you experiment with new recipes?\n"
        "9. Share a story about cooking for a special occasion.\n"
        "10. How do you adapt recipes to fit dietary preferences?\n"
        "11. Describe your kitchen essentials.\n"
        "12. What's your go-to comfort food to cook?\n"
        "13. How do you involve others in cooking?\n"
        "14. Share a cooking tradition in your family.\n"
        "15. What's your favorite kitchen gadget?\n"
        "16. How do you find inspiration for new dishes?\n"
        "17. Describe a cultural dish you love to prepare.\n"
        "18. What's your strategy for meal prepping?\n"
        "19. Share a cooking challenge you've overcome.\n"
        "20. How do you celebrate through cooking?\n\n"
        "Week 4\n"
        "Day 25: Technology\n"
        "1. How has technology changed your daily life?\n"
        "2. Describe your favorite tech gadget and its benefits.\n"
        "3. How do you stay updated with the latest technology news?\n"
        "4. Share a story about a technological advancement you find fascinating.\n"
        "5. What's your favorite app and why?\n"
        "6. How do you use technology for learning or productivity?\n"
        "7. Describe a time when technology helped you in an unexpected way.\n"
        "8. How do you feel about the impact of social media on society?\n"
        "9. Share a technology-related prediction you have for the future.\n"
        "10. How do you protect your privacy online?\n"
        "11. Describe a tech-related problem you've solved.\n"
        "12. What's your opinion on artificial intelligence?\n"
        "13. How do you help others with technology challenges?\n"
        "14. Share a story about a memorable tech support experience.\n"
        "15. How do you balance screen time with offline activities?\n"
        "16. Describe your first experience using a computer.\n"
        "17. What tech skill do you want to improve?\n"
        "18. How do you approach learning new technology?\n"
        "19. Share a technological innovation you're excited about.\n"
        "20. How do you think technology will impact education in the future?\n\n"
        "Day 26: Restaurants\n"
        "1. How do you discover new restaurants in your area?\n"
        "2. Describe your favorite cuisine and why you love it.\n"
        "3. Share a story about a memorable dining experience.\n"
        "4. How do you choose where to eat when traveling?\n"
        "5. What's your favorite dish to order at restaurants?\n"
        "6. Describe a favorite restaurant atmosphere.\n"
        "7. How do you find restaurants that cater to dietary restrictions?\n"
        "8. Share a dining experience where the service exceeded your expectations.\n"
        "9. How do you support local restaurants?\n"
        "10. Describe a restaurant you would recommend to others.\n"
        "11. What's your approach to tipping at restaurants?\n"
        "12. Share a story about a restaurant you visit regularly.\n"
        "13. How do you celebrate special occasions at restaurants?\n"
        "14. Describe a restaurant that holds sentimental value for you.\n"
        "15. How do you feel about food delivery services?\n"
        "16. Share a restaurant where you celebrated a milestone.\n"
        "17. How do you find hidden gem restaurants?\n"
        "18. Describe a favorite dessert you've had at a restaurant.\n"
        "19. What's your opinion on themed restaurants?\n"
        "20. How do you choose restaurants for group outings?\n\n"
        "Day 27: Health and Fitness\n"
        "1. How do you prioritize your health and fitness goals?\n"
        "2. Share a fitness accomplishment you're proud of.\n"
        "3. Describe your favorite type of exercise and why.\n"
        "4. How do you motivate yourself to stay active?\n"
        "5. Share a story about a fitness challenge you participated in.\n"
        "6. What's your approach to maintaining a balanced diet?\n"
        "7. How do you manage stress through exercise?\n"
        "8. Describe a fitness routine that works for you.\n"
        "9. Share a tip for staying hydrated throughout the day.\n"
        "10. How do you unwind after a tough workout?\n"
        "11. Describe a wellness trend you're curious about.\n"
        "12. What's your favorite outdoor activity for fitness?\n"
        "13. How do you set and track fitness goals?\n"
        "14. Share a story about a health or fitness transformation.\n"
        "15. How do you stay motivated during your fitness journey?\n"
        "16. Describe your morning or evening stretch routine.\n"
        "17. What's your approach to recovery after exercise?\n"
        "18. How do you balance exercise with a busy schedule?\n"
        "19. Share a fitness resource or app you find helpful.\n"
        "20. How do you encourage friends or family to prioritize health?\n\n"
        "Day 28: Dream Job\n"
        "1. Describe your dream job and why it appeals to you.\n"
        "2. How do you prepare for a job interview?\n"
        "3. Share a story about a career milestone you achieved.\n"
        "4. How do you set career goals and track your progress?\n"
        "5. What's your approach to networking in your field?\n"
        "6. Describe a mentor who has influenced your career.\n"
        "7. How do you handle career setbacks or challenges?\n"
        "8. Share a lesson you've learned from a job experience.\n"
        "9. What's your strategy for professional development?\n"
        "10. How do you stay current with industry trends?\n"
        "11. Describe a career change you're considering.\n"
        "12. How do you negotiate salary and benefits?\n"
        "13. Share a story about a job opportunity you pursued.\n"
        "14. How do you balance work and personal life in your career?\n"
        "15. What skills do you want to develop further?\n"
        "16. Describe a job that taught you unexpected skills.\n"
        "17. How do you define success in your career?\n"
        "18. Share a story about a job that shaped your perspective.\n"
        "19. How do you stay positive during job searches?\n"
        "20. What advice would you give to someone pursuing their dream job?\n\n"
        "Day 29: A Special Memory\n"
        "1. Share a childhood memory that shaped who you are today.\n"
        "2. Describe a memorable family gathering or celebration.\n"
        "3. How did you celebrate a significant life milestone?\n"
        "4. Share a story about a cherished possession.\n"
        "5. Describe a tradition or ritual that's important to you.\n"
        "6. How do you commemorate anniversaries or special dates?\n"
        "7. Share a memory of a spontaneous adventure.\n"
        "8. Describe a meaningful conversation you've had.\n"
        "9. How has a special memory influenced your beliefs?\n"
        "10. Share a lesson you learned from a challenging experience.\n"
        "11. Describe a memory that makes you smile.\n"
        "12. How do you share memories with loved ones?\n"
        "13. Share a story about a surprise you've experienced.\n"
        "14. Describe a tradition you hope to pass on.\n"
        "15. How do you keep memories alive over time?\n"
        "16. Share a story about a memorable journey or trip.\n"
        "17. Describe a memory that inspires gratitude.\n"
        "18. How do you celebrate personal achievements?\n"
        "19. Share a memory that taught you resilience.\n"
        "20. How do you reflect on important life moments?\n\n"
        "Day 30: Your Neighborhood\n"
        "1. How do you contribute to your community?\n"
        "2. Describe a community event you've participated in.\n"
        "3. How do you support local businesses in your area?\n"
        "4. Share a story about a neighborhood improvement project.\n"
        "5. Describe a neighborhood tradition or festival.\n"
        "6. How do you promote environmental awareness locally?\n"
        "7. Share a story about a neighbor who inspired you.\n"
        "8. Describe a local charity you support.\n"
        "9. How do you celebrate diversity in your community?\n"
        "10. Share a story about a local hero or role model.\n"
        "11. Describe a neighborhood park or gathering spot you enjoy.\n"
        "12. How do you connect with neighbors?\n"
        "13. Share a memory of a community celebration.\n"
        "14. Describe a local business you recommend.\n"
        "15. How has your neighborhood changed over time?\n"
        "16. Share a story about a community initiative you've been part of.\n"
        "17. How do you advocate for local issues?\n"
        "18. Describe a neighborhood tradition you look forward to.\n"
        "19. How do you collaborate with others in your community?\n"
        "20. Share a lesson you've learned from your neighborhood.\n"
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
