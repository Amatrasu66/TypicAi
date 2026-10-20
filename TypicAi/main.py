
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise SystemExit("ERROR: NVIDIA_API_KEY not found in .env")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=0,
)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Give clear, useful answers."
    }
]

print("GPT-OSS-20B Chat")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()

        if user_input.lower() in ("exit", "quit"):
            print("Chat ended.")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nAssistant: ", end="", flush=True)

        stream = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=1,
            top_p=1,
            max_tokens=1024,
            stream=True,
        )

        answer_parts = []

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                print(content, end="", flush=True)
                answer_parts.append(content)

        answer = "".join(answer_parts)
        messages.append({"role": "assistant", "content": answer})

        print("\n")

    except KeyboardInterrupt:
        print("\nChat ended.")
        break

    except Exception as error:
        print(f"\nAPI error: {type(error).__name__}: {error}")
        if messages and messages[-1]["role"] == "user":
            messages.pop()
        print("You can try again.\n")
