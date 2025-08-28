from langfuse import Langfuse

langfuse = Langfuse()

# ---------- Q&A Prompt ----------
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
        "model": "gpt-4o",
        "temperature": 0.2
    }
)

# ---------- Summarizer Prompt ----------
langfuse.create_prompt(
    name="search-summarizer",
    type="text",
    prompt="""Summarize the following results:

{% for r in results %}
{{ r.title }}: {{ r.content }}
{% endfor %}
""",
    labels=["production", "summarizer"],
    config={
        "model": "gpt-4o-mini",
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
            "content": """You are a routing agent.
Decide which tool to use for a given question:
- If the question is about Budget Speech 2024-2025, call `retrieval_tool`.
- Otherwise, call `search_tool`.
You must always use one of these tools, never answer directly yourself."""
        },
        {"role": "human", "content": "{{input}}"},
        {"role": "system", "content": "Available tools:\n{{tools}}\n\nRemember: use only {{tool_names}}."},
        {"role": "system", "content": "Previous steps:\n{{agent_scratchpad}}"}
    ],
    labels=["production", "router"],
    config={
        "model": "gpt-4o",
        "temperature": 0.0
    }
)
