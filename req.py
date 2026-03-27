import requests

url = "https://api.elevenlabs.io/v1/voices"

headers = {
    "xi-api-key": "sk_866d22f516d4d2587824743b4c438aa7649f139662b185d2"
}

resp = requests.get(url, headers=headers)
print(resp.json())