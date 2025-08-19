# create_dataset.py
import ollama
import json
import random
import pandas as pd
import requests

# --- Configuration ---
OLLAMA_MODEL = 'llama3'
NUM_EXAMPLES_TO_GENERATE = 100
OUTPUT_FILE = 'stock_advisor_dataset.jsonl'

def get_sp500_companies():
    """Fetches the list of S&P 500 companies and their symbols from Wikipedia."""
    print("Fetching latest S&P 500 company list from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        response = requests.get(url, headers={'User-Agent': 'MyCoolBot/0.0 (https://example.org/bot; my@email.com)'})
        response.raise_for_status()
        sp500_table = pd.read_html(response.text)[0]
        print("Successfully fetched company list.")
        return sp500_table[['Symbol', 'Security']]
    except Exception as e:
        print(f"Could not fetch S&P 500 list: {e}. Falling back to a small, fixed list.")
        fallback_data = {"Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL"}
        return pd.DataFrame(list(fallback_data.items()), columns=['Security', 'Symbol'])

INTENTS = ["stock_price", "company_news", "market_sentiment", "company_outlook"]

def generate_synthetic_data_point(companies_df):
    """Generates a single, high-quality training example using a powerful LLM."""
    random_company = companies_df.sample(1).iloc[0]
    company_name = random_company['Security']
    symbol = random_company['Symbol']
    intent = random.choice(INTENTS)
    
    prompt_for_generator = f"""
    You are a data generation assistant. Your task is to generate a single, high-quality conversational exchange for a stock advisor chatbot.
    **Instructions:**
    1. Create a realistic user question (`prompt`).
    2. Create a corresponding ideal chatbot answer (`response`).
    3. The answer should be conversational, helpful, and embody the persona of a friendly stock market expert.
    4. The answer MUST include a subtle disclaimer about not being financial advice.
    5. Format the output as a single, raw JSON object with two keys: "prompt" and "response". Do not include any other text or markdown formatting like ```json.

    **Data for this example:**
    - Company: {company_name} ({symbol})
    - User Intent: {intent}
    """
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt_for_generator}],
            format="json"
        )
        
        # **IMPROVED ERROR HANDLING**
        content = response['message']['content']
        return json.loads(content)
        
    except json.JSONDecodeError:
        print(f"  -> ERROR: Ollama did not return valid JSON. Response was:\n---\n{content}\n---")
        return None
    except Exception as e:
        print(f"  -> ERROR: An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    companies_df = get_sp500_companies()
    generated_count = 0
    
    print(f"\nStarting dataset generation with '{OLLAMA_MODEL}'...")
    
    with open(OUTPUT_FILE, 'w') as f:
        for i in range(NUM_EXAMPLES_TO_GENERATE):
            print(f"Generating example {i + 1}/{NUM_EXAMPLES_TO_GENERATE}...")
            data_point = generate_synthetic_data_point(companies_df)
            
            if data_point and 'prompt' in data_point and 'response' in data_point:
                formatted_example = {
                    "text": f"<s>[INST] {data_point['prompt']} [/INST] {data_point['response']} </s>"
                }
                f.write(json.dumps(formatted_example) + '\n')
                generated_count += 1
            else:
                print("  -> Failed to generate a valid data point. Skipping.")

    print(f"\nDataset generation complete. Successfully saved {generated_count} examples to '{OUTPUT_FILE}'.")

