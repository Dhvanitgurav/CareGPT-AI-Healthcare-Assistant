import streamlit as st
import time
import speech_recognition as sr
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
medical_model = AutoModelForCausalLM.from_pretrained(model_name)


if "messages" not in st.session_state:
    st.session_state.messages = []

def is_medical_query(user_input):
    medical_keywords = ["symptom", "medicine", "disease", "doctor", "treatment", "fever", "cold", "pain", "infection", "prescription", "health", "surgery"]
    return any(keyword in user_input.lower() for keyword in medical_keywords)

def healthcare_chatbot(user_input):
    user_input = user_input.lower()
    predefined_responses = {
        "symptom": "⚠️ Please consult a doctor for an accurate diagnosis.",
        "appointment": "📅 Would you like to schedule an appointment with the doctor?",
        "medication": "💊 It's important to take prescribed medicines regularly."
    }
    for keyword, response in predefined_responses.items():
        if keyword in user_input:
            return response
    
    if is_medical_query(user_input):
        prompt = f"Patient query: {user_input}\nDoctor response:"
        inputs = tokenizer(prompt, return_tensors="pt")
        output = medical_model.generate(**inputs, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2)
        return tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    return "I'm sorry, I didn't understand that. Can you provide more details?"

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = recognizer.listen(source)
        st.info("⏳ Processing...")
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "⚠️ Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "⚠️ Error with speech recognition service."

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text if text.strip() else "⚠️ No readable text found in the image."

st.set_page_config(page_title="CareGPT - Medical AI", layout="wide")
st.sidebar.title("📜 Chat History")

for msg in st.session_state.messages:
    st.sidebar.write(f"**{msg['role'].capitalize()}:** {msg['content']}")

st.title("💬 CareGPT - Your Medical AI Assistant")

if st.button("🎤 Speak"):
    voice_text = speech_to_text()
    if voice_text:
        st.write(f"🗣 **You said:** {voice_text}")
        st.session_state.messages.append({"role": "user", "content": voice_text})
        response = healthcare_chatbot(voice_text)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(f"**CareGPT:** {response}")

uploaded_image = st.file_uploader("📷 Upload a medical image", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    extracted_text = extract_text_from_image(image)
    st.write("📜 **Extracted Text:**")
    st.text(extracted_text)
    if extracted_text != "⚠️ No readable text found in the image.":
        response = healthcare_chatbot(extracted_text)
        with st.chat_message("assistant"):
            st.markdown(f"**CareGPT:** {response}")

user_input = st.chat_input("How can I assist you today?")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(f"**User:** {user_input}")
    with st.spinner("Thinking..."):
        response = healthcare_chatbot(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(f"**CareGPT:** {response}")
