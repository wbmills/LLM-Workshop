import openai
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_KEY")
openai.api_key = api_key

def query_gpt(text, query):
    """Sends text to GPT and extracts relevant insights."""
    prompt = f"""
    The following text is extracted from a research paper on ecosystems:

    {text}

    Based on this, answer the following query:
    {query}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response["choices"][0]["message"]["content"]

# Example usage
query = "What factors indicate a healthy or unhealthy ecosystem?"
insights = query_gpt(extracted_text, query)
print(insights)
