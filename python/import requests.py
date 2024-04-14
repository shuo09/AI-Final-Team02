import requests

def call_chatgpt(prompt):
    api_url = "https://api.openai.com/v1/engines/davinci-codex/completions"
    headers = {
        "Authorization": f"Bearer YOUR_API_KEY",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "max_tokens": 150
    }
    response = requests.post(api_url, json=data, headers=headers)
    return response.json()

prompt = "Explain blockchain technology"
response = call_chatgpt(prompt)
print(response)