import streamlit as st
import requests
import json
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Stock Advisory Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Use session state to manage the backend URL, making it accessible globally
if 'backend_url' not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:5000/api/chat"

# --- Central Query Handler ---
def handle_query(prompt):
    """
    Handles the query by updating the session state with the user's message
    and the bot's response, then triggers a rerun to update the UI.
    """
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get assistant response
    try:
        payload = {"messages": st.session_state.messages}
        response = requests.post(
            st.session_state.backend_url,  # Use the URL from session state
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        bot_response = response.json().get("response", "Sorry, something went wrong.")
    except requests.exceptions.RequestException as e:
        bot_response = f"Connection Error: {e}"
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Rerun the app to show the new messages
    st.rerun()

# --- UI Styling ---
st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .stChatMessage { border-radius: 10px; padding: 1rem; margin-bottom: 1rem; }
    .stChatMessage[data-testid="stChatMessageContent"] {
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
        /* --- NEW & IMPROVED Button Styles --- */
    .stButton>button {
        font-size: 8px;
        padding: 6px 12px;
        margin: 0px 5px;
        border-radius: 20px;
        background-color: #e8eef2;
        border: none;
        color: #3a5a78;
        font-weight: 500;
        transition: all 0.2s ease; /* Smooth transition for hover */
    }
    .stButton>button:hover {
        background-color: #d0dce4;
        color: #274057;
    }
    .stButton>button:focus {
        outline: none;
        box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25); /* Subtle focus ring */
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.title("Stock Advisory System")
    st.markdown("---")
    # This text input now updates the backend URL stored in the session state
    st.session_state.backend_url = st.text_input("Backend API URL", st.session_state.backend_url)
    st.markdown("---")
    st.header("Features")
    st.markdown("""
    - **Real-time Stock Information**
    - **Market Trend Analysis**
    - **Simulated Price Prediction**
    - **Latest Company News**
    """)
    st.markdown("---")
    
    # --- Clickable Sample Queries ---
    st.header("Sample Queries")
    if st.button("Hello"):
        handle_query("Hello")
    if st.button("What is the price of Apple?"):
        handle_query("What is the price of Apple?")
    if st.button("Predict the price of GOOGL"):
        handle_query("Predict the price of GOOGL")
        
    st.markdown("---")
    st.warning("**Disclaimer**: Educational tool only. Not financial advice.")

# --- Main Chat Interface ---
st.title("AI Financial Chatbot 🤖")

# Initialize chat history with a welcome message
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI Stock Advisor. How can I help you today?"}]

# Display all messages from chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

  
# Handle user input from the chat box
if prompt := st.chat_input("What would you like to know?"):
    handle_query(prompt)

