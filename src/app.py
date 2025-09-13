# app.py
import streamlit as st
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import your backend functions
from backend.data_fetcher import get_stock_price, get_company_news
from models.lstm_predictor import predict_next_price
from backend.response_generator import generate_response
from backend.spacy_nlu_processor import get_intent_and_entities, train_spacy_nlu_model, load_spacy_nlu_model
from backend.config import SPACY_MODEL_DIR

# --- Model Loading ---
@st.cache_resource
def load_model():
    if not os.path.exists(SPACY_MODEL_DIR):
        train_spacy_nlu_model()
    return load_spacy_nlu_model()

load_model()

# User input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            latest_message = st.session_state.messages[-1]['content']
            history = st.session_state.messages[:-1]

            # 1. Process NLP
            intent, symbol = get_intent_and_entities(latest_message, history)
            raw_data = None

            # 2. Fetch data based on intent
            if intent == "greeting":
                raw_data = {"response": "Hello! How can I assist you with your stock market questions today?"}
            # ... (other intents)

            # 3. Generate response
            response_text = generate_response(intent, raw_data, latest_message, history)

        st.markdown(response_text, unsafe_allow_html=True)