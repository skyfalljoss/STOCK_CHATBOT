# frontend/streamlit_app.py - Streamlit Frontend
import streamlit as st
import requests
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Advisory Chatbot",
    page_icon="🤖",
    layout="wide"
)

# --- UI Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .stChatMessage { border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .stChatMessage[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("Stock Advisory System")
    st.markdown("---")
    backend_url = st.text_input("Backend API URL", "http://127.0.0.1:5000/api/chat")
    st.markdown("---")
    st.header("Features")
    st.markdown("""
    - **Real-time Stock Information**
    - **Market Trend Analysis**
    - **Simulated Price Prediction**
    - **Latest Company News**
    """)
    st.markdown("---")
    st.header("Sample Queries")
    st.info("What is the price of Apple?")
    st.info("Predict the price of GOOGL")
    st.markdown("---")
    st.warning("**Disclaimer**: Educational tool only. Not financial advice.")

# --- Main Chat Interface ---
st.title("AI Financial Chatbot 🤖")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI Stock Advisor. How can I help you today?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# User input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {"messages": st.session_state.messages}
                response = requests.post(
                    backend_url,
                    headers={"Content-Type": "application/json"},
                    # data=json.dumps({"message": prompt})
                    data=json.dumps(payload)
                )
                response.raise_for_status()
                bot_response = response.json().get("response", "Sorry, something went wrong.")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the backend. Please ensure it's running.")
                bot_response = f"Connection Error: {e}"
        
        st.markdown(bot_response, unsafe_allow_html=True)
    
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
