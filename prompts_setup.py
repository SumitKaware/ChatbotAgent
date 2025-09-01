import os
from dotenv import load_dotenv
from langfuse import Langfuse
from model import gemini_model

# Load environment variables from .env file
load_dotenv()

# Initialize Langfuse client
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
langfuse_host = os.getenv("LANGFUSE_HOST", "")

if not langfuse_secret_key or not langfuse_public_key or not langfuse_host:
    raise ValueError("Langfuse credentials are not set in the environment.")

langfuse = Langfuse(
  secret_key=langfuse_secret_key,
  public_key=langfuse_public_key,
  host=langfuse_host
)

langfuse.create_prompt(
    name="qa-contextual",
    type="text",
    prompt="""You are a helpful AI assistant. Answer the user's question based on the provided context and chat history.
Improve the quality of the answer by using the context provided and the chat history.
If any calculations are needed, do them step by step and explain your reasoning.
If the answer is not in the context, state that you don't know.

Chat History:
{{chat_history}}

Context:
{{context}}

Question: {{question}}
Answer:""",
    labels=["production", "qa"],
    config={
        "model": gemini_model,
        "temperature": 0.2
    }
)

# ---------- Summarizer Prompt ----------
langfuse.create_prompt(
    name="search-summarizer",
    type="text",
    prompt="""Summarize the following results:{results_dict}""",
    labels=["production", "summarizer"],
    config={
        "model": gemini_model,
        "temperature": 0.3
    }
)

# ---------- Router Agent Prompt ----------
langfuse.create_prompt(
    name="router-agent",
    type="chat",
    prompt=[
        {
            "role": "system",
            "content": """You are a routing agent. Your task is to decide which tool to use for a given question.
- If the question is about Budget Speech 2024-2025, use the `retrieval_tool`.
- Otherwise, use the `search_tool`.
You must choose one of these tools.
"""
        },
        {"role": "human", "content": "{input}"}
    ],
    labels=["production", "router"],
    config={
        "model": gemini_model,
        "temperature": 0.0
    }
)
