# backend/rasa_nlu_processor.py
import asyncio
from rasa.core.agent import Agent
from rasa.model_training import train_nlu
import os

# --- Configuration ---
NLU_MODEL_DIR = "./rasa_nlu/models"
NLU_DATA_FILE = "./rasa_nlu/nlu.yml"
CONFIG_FILE = "./rasa_nlu/config.yml" # We will create this file next

# Global agent to hold the trained model
agent = None

async def train_and_load_model():
    """
    Trains a new Rasa NLU model if one doesn't exist, then loads it.
    """
    global agent
    
    # Check if a trained model exists
    if not os.path.exists(NLU_MODEL_DIR) or not os.listdir(NLU_MODEL_DIR):
        print("No trained Rasa NLU model found. Training a new one...")
        await train_nlu(
            nlu_data=NLU_DATA_FILE,
            config=CONFIG_FILE,
            output=NLU_MODEL_DIR
        )
        print("Rasa NLU model trained successfully.")

    # Load the trained model
    model_path = os.path.join(NLU_MODEL_DIR, os.listdir(NLU_MODEL_DIR)[0])
    agent = Agent.load(model_path)
    print(f"Rasa NLU model loaded from {model_path}")

def get_intent_and_entities(text, history=None):
    """
    Parses the user's text using the trained Rasa NLU model.
    """
    if not agent:
        raise RuntimeError("Rasa agent is not loaded. Run train_and_load_model() first.")

    # Parse the message
    result = asyncio.run(agent.parse_message(message_data={'text': text}))
    
    intent = result.get('intent', {}).get('name', 'unknown')
    entities = result.get('entities', [])
    
    symbol = None
    company_map = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
        'alphabet': 'GOOGL', 'amazon': 'AMZN', 'tesla': 'TSLA',
        'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA'
    }

    if entities:
        # Prioritize entities found by Rasa
        entity = entities[0]
        value = entity['value'].lower()
        if entity['entity'] == 'symbol':
            symbol = value.upper()
        elif entity['entity'] == 'company':
            if value in company_map:
                symbol = company_map[value]
            elif f"{value} stocks" in company_map: # Handle cases like "google stocks"
                 symbol = company_map[f"{value} stocks"]
    
    # Fallback: search history if no entity found in current message
    if not symbol and history:
        for message in reversed(history):
            # A simpler text search is sufficient for history context
            for name, ticker in company_map.items():
                if name in message['content'].lower():
                    symbol = ticker
                    print(f"Found context symbol '{symbol}' in history.")
                    break
            if symbol:
                break
                
    return intent, symbol
