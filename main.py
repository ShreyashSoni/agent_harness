import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def main():
    print("Hello from agent-harness!")

    client = OpenAI(
        base_url="https://floodgate.g.apple.com/api/openai/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    response = client.chat.completions.create(
        model="anthropic.claude-sonnet-5",
        messages=[{"role": "user", "content": "Say 'setup works' and nothing else."}]
    )

    return response.choices[0].message.content




if __name__ == "__main__":
    response = main()
    print(response)
