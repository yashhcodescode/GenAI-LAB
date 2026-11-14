import os
from dotenv import load_dotenv
from google import genai

# Step 1: Load API key from .env
load_dotenv()

# Step 2: Read Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Gemini API key not found!")

# Step 3: Initialize Gemini client
client = genai.Client(api_key=api_key)

# Step 4: Make first LLM call
model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
try:
    response = client.models.generate_content(
        model=model_name,
        contents="What is Artificial Intelligence? Explain in simple words."
    )
    # Step 5: Display response
    print("\nLLM Response:")
    print(response.text)
except Exception as e:
    # If preferred model experiences high demand (e.g. 503 on gemini-3.8-flash), fallback to gemini-3.5-flash-lite
    if model_name != "gemini-3.5-flash-lite":
        print(f"Notice: {model_name} is busy/unavailable ({e}). Trying gemini-3.5-flash-lite...")
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents="What is Artificial Intelligence? Explain in simple words."
        )
        print("\nLLM Response:")
        print(response.text)
    else:
        raise
