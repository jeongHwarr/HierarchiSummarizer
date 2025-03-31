import time
import requests

import ollama
import openai
from mistralai import Mistral as Mixtral_client
from mistralai.models.sdkerror import SDKError

def make_messages(system_prompt, prompt):
    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})

    messages.append({'role': 'user', 'content': prompt})
    return messages

class OpenAI:
    def __init__(self, model, api_key, temperature, system_prompt):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.system_prompt = system_prompt

    def get_response(self, prompt):
        openai.api_key = self.api_key

        messages = make_messages(self.system_prompt, prompt)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()


class Groq:
    def __init__(self, model, api_key, temperature, system_prompt):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.system_prompt = system_prompt

    def get_response(self, prompt):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"
        }

        messages = make_messages(self.system_prompt, prompt)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        response = requests.post(url, headers=headers, json=payload)
        try:
            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            print(e)

class Ollama():
    def __init__(self, model, base_url, temperature, system_prompt):
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    def get_response(self, prompt):
        messages = []

        messages = make_messages(self.system_prompt, prompt)

        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={'temperature': self.temperature}
        )
        return response['message']['content']

class Mistral:
    def __init__(self, model, api_key, temperature, system_prompt):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.system_prompt = system_prompt

    def get_response(self, prompt):
        max_retries = 3
        retry_delay = 1

        client = Mixtral_client(api_key=self.api_key)
        messages = make_messages(self.system_prompt, prompt)

        for attempt in range(max_retries):
            try:
                chat_response = client.chat.complete(
                    model= self.model,
                    messages = messages,
                    temperature = self.temperature,
                )
                return chat_response.choices[0].message.content
            except SDKError as e:
                if "rate limit" in str(e).lower():
                    print(f"Retry...{e}")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise
