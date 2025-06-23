import streamlit as st
import io
import os
import requests
import json
from gtts import gTTS # Google Text-to-Speech
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from io import BytesIO

# --- Configuration ---
# Your Google API Key for Gemini.
# In Canvas, leave this empty; it will be injected at runtime.
# For local execution, set it as an environment variable or hardcode (not recommended for production).
api_key = "api_key_here"

# --- LLM and Speech-to-Text/Text-to-Speech Models ---
# Using gemini-2.0-flash for conversational tasks.
GEMINI_LLM_MODEL = "gemini-2.0-flash"
# For embeddings, often used in RAG, but not directly in this simple chat.
GEMINI_EMBEDDING_MODEL = "models/embedding-001"

# --- Utility Functions ---

def get_gemini_response(chat_history_messages):
    """
    Calls the Gemini API to get a conversational response.
    Args:
        chat_history_messages (list): A list of dictionaries representing the chat history.
                                    [{'role': 'user', 'parts': [{'text': '...'}]}, ...]
    Returns:
        str: The generated text response from Gemini.
    """
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_LLM_MODEL}:generateContent?key={api_key}"
    
    # LangChain ChatGoogleGenerativeAI handles history directly
    llm = ChatGoogleGenerativeAI(model=GEMINI_LLM_MODEL, temperature=0.7, google_api_key=api_key)
    
    # Convert chat history into LangChain's HumanMessage/AIMessage format
    lc_messages = []
    for msg in chat_history_messages:
        if msg['role'] == 'user':
            lc_messages.append(HumanMessage(content=msg['parts'][0]['text']))
        elif msg['role'] == 'model': # Gemini's role is 'model'
            lc_messages.append(AIMessage(content=msg['parts'][0]['text']))
            
    try:
        # Use LangChain's invoke method for simplicity and recommended practice
        response = llm.invoke(lc_messages)
        raw_text = response.content
        # Clean up common Markdown formatting (e.g., asterisks for bold)
        cleaned_text = raw_text.replace('**', '').replace('*', '') # Remove bolding asterisks
        return cleaned_text
    except Exception as e:
        st.error(f"Error calling Gemini LLM: {e}")
        return "I'm sorry, I couldn't get a response from the AI at the moment."

def text_to_speech(text):
    """
    Converts text to speech using Google Text-to-Speech (gTTS).
    Args:
        text (str): The text to convert.
    Returns:
        bytes: Audio data in MP3 format.
    """
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = BytesIO()
        # Changed from tts.save(audio_fp) to tts.write_to_fp(audio_fp)
        # as save() expects a file path, while write_to_fp() expects a file-like object.
        tts.write_to_fp(audio_fp) 
        audio_fp.seek(0) # Rewind the BytesIO object to the beginning
        return audio_fp.read()
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Voice-Controlled AI Chat", page_icon="🎙️")
    st.title("🎙️ Voice-Controlled AI Chat")
    st.markdown("Speak to the AI, and it will respond both in text and audio!")

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Stores raw message dictionaries for Gemini API

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["parts"][0]["text"])
            # If it's an AI message, also play the audio
            if message["role"] == "model" and "audio" in message:
                st.audio(message["audio"], format='audio/mp3', start_time=0)

    # Microphone input
    audio_input = st.chat_input("Speak to the AI...")

    if audio_input:
        # --- 1. Speech-to-Text (Conceptual/Simulated) ---
        # In a real app, this would involve a speech-to-text API call (e.g., Google Speech-to-Text, Whisper API)
        # For this demonstration, we're taking text input directly as if it came from STT.
        user_text = audio_input # User types their "speech" for simplicity in this demo

        # Add user's text to chat history
        st.session_state.chat_history.append({"role": "user", "parts": [{"text": user_text}]})
        with st.chat_message("user"):
            st.write(user_text)

        # --- 2. Text Generation (LLM) ---
        with st.spinner("AI is thinking..."):
            ai_response_text = get_gemini_response(st.session_state.chat_history)
            
            # Add AI's text response to chat history immediately
            st.session_state.chat_history.append({"role": "model", "parts": [{"text": ai_response_text}]})
            with st.chat_message("assistant"):
                st.write(ai_response_text)

        # --- 3. Audio Response (Text-to-Speech) ---
        with st.spinner("Generating audio..."):
            ai_audio = text_to_speech(ai_response_text)
            if ai_audio:
                # Store audio in history for replay on refresh/re-run
                st.session_state.chat_history[-1]["audio"] = ai_audio 
                st.audio(ai_audio, format='audio/mp3', start_time=0)

if __name__ == "__main__":
    main()
