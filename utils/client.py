import os
import openai
import httpx

api_key = os.environ.get('OPENAI_API_KEY')
api_base = "http://localhost:4000/v1"

def client(api_key=api_key,base_url=api_base):
    return openai.OpenAI(api_key=api_key, base_url=api_base)
