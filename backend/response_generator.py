# backend/response_generator.py
import ollama

# --- Configuration ---
# Specify the model you have running in Ollama (e.g., 'llama3', 'deepseek-coder', 'gemini')
OLLAMA_MODEL = 'llama3' 

def generate_response(intent, data, user_query, history):
    """
    Generates a natural language response using a local LLM via Ollama.
    """
    prompt = create_prompt(intent, data, user_query, history)
    # The history is now part of the prompt itself, so we send it all
    messages_for_ollama = history + [{'role': 'user', 'content': prompt}]
    
    # # If the prompt is just an error message, return it directly
    # if data and data.get("error"):
    #     return data["error"]

    try:
        # Send the prompt to the Ollama API
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages_for_ollama,
            stream=False
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return f"I'm having trouble connecting to my AI brain ({OLLAMA_MODEL}). Is Ollama running?"

def create_prompt(intent, data, user_query, history):
    """
    Creates a detailed prompt for the LLM based on the user's intent and fetched data.
    """
    # System message to define the AI's personality
    system_prompt = "You are a friendly, conversational, and knowledgeable stock market advisory chatbot. Your goal is to provide clear, helpful, and natural-sounding information, not just raw data. Never give direct financial advice, and always include a subtle disclaimer that your information is for educational purposes(Highlight or Bold the disclaimer). Write it in a way that feels like a conversation with a human friend who is knowledgeable about stocks and finance. And answer consicely, no long paragraphs. Make sure write the correct data that you fetched from the backend, do not make up any data or information."


    # If the data dictionary contains an error, create a special prompt to handle it gracefully.
    # if data and data.get("error"):
    #     error_message = data["error"]
    #     prompt = f"""{system_prompt}

    #     A user's query "{user_query}" resulted in an error: "{error_message}".
        
    #     Your task is to explain this error to the user in a helpful and conversational way.
    #     - Do NOT just repeat the error message.
    #     - If the error is about a ticker not being found (like "Could not retrieve data"), suggest that it might be a typo and offer a potential correction if it seems obvious (e.g., 'NVDIA' -> 'NVDA').
    #     - Politely ask them to clarify or try again.
    #     - Answer concisely, no long paragraphs make it easy to focus on the information.
    #     """
    #     return prompt

    if data and data.get("error"):
        error_message = data["error"]
        return f"""A user's query "{user_query}" resulted in an error: "{error_message}". Explain this error to the user in a helpful and conversational way. If the error is about a ticker not being found, suggest it might be a typo. Answer concisely, no long paragraphs make it easy to focus on the information."""

    if not data:
        # Handle general intents without specific data
        if intent == "greeting":
            return f"{system_prompt}\n\nThe user said: '{user_query}'. Respond with a friendly greeting."
        elif intent == "market trend":
            return "The user asked about market trends. Explain that trends are complex and depend on major indices, then offer to look up a specific stock."
        else:
            return f"{system_prompt}\n\nThe user said: '{user_query}'. Based on the conversation history, provide a helpful response or ask for clarification."
    # Handle intents with fetched data
    if intent == 'stock price':
        return f"The user asked about a stock price. You have new data: the price for {data.get('symbol', 'N/A')} is ${data.get('price', 0.0):.2f}. Incorporate this new information into a friendly response. Make sure the answer is concise and easy to read, not just a data dump. Highlight or bold the price information for emphasis."

    if intent == 'company news':
        headlines = "\n- ".join(article['title'] for article in data.get('articles', []))
        print(headlines)
        return f"The user asked for news about {data.get('symbol', 'N/A')}. You have found these headlines:\n- {headlines}\nSummarize this news conversationally. Provide the URLs for more details, but don't just list them. Make it feel like a conversation, not a data dump. Make sure the answer is concise and easy to read."

    if intent == 'prediction':
        return f"The user asked for a price prediction for {data.get('symbol', 'N/A')}. Your internal model's analysis is: Trend is {data.get('trend', 'unknown')} and projected price is around ${data.get('projected_price', 0.0):.2f}. Explain this simulated prediction, strongly emphasizing it's not financial advice. Make sure the answer is concise and easy to read, not just a data dump. Highlight or bold the projected price for emphasis."
        
    return f"The user said: '{user_query}'. Respond helpfully based on the conversation history."

