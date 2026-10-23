"""List the chat model IDs your OpenAI key can access (model names only).

Usage: python -m scripts.list_openai_models
"""

import os

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    load_dotenv()
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model_ids = sorted(model.id for model in client.models.list())
    chat_like = [m for m in model_ids if m.startswith(("gpt-", "o"))]
    print(f"{len(chat_like)} chat-family models available:")
    for model_id in chat_like:
        print(" ", model_id)


if __name__ == "__main__":
    main()