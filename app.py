import os
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.tools import Tool
from model import tavily_search
# from langgraph.prebuilt import AgentState
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import ChatPromptTemplate
from model import llm  # Importing the LLM and embeddings from model.py
from pdfloader import vector_db
from storing_embeddings import load_chroma_db_and_retriever
from langfuse import Langfuse
import json

langfuse = Langfuse()
#from langchain_tavily import TavilySearch  # Importing raw documents from pdfloader.py
#
# Create a vector database

#vector_db()  # Initialize the vector store with a sample query

# --- Demonstrate loading ChromaDB and using retriever ---
print("\n--- Demonstrating ChromaDB Loading and Retrieval ---")
current_script_dir = os.getcwd()
output_embedding_directory = os.path.join(current_script_dir, "embedded_content")
sample_chroma_path = os.path.join(output_embedding_directory, "chroma_db")
print(f"Loading ChromaDB from: {sample_chroma_path}")

rag_retriever = load_chroma_db_and_retriever(sample_chroma_path)
print("Retriever loaded:", rag_retriever)

# --- 3. LangGraph Agent Definition ---

# Define the state of our graph
class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's question.
        chat_history: A list of messages forming the conversation history.
        documents: A list of retrieved documents.
        generation: The generated response from the LLM.
    """
    question: str
    chat_history: Annotated[List[BaseMessage], operator.add]
    documents: List[Document]
    generation: str
    search_results: dict
    summary: dict
    decision: str


def retrieve(state: AgentState):
    """
    Retrieves documents based on the user's question.
    """
    print("---RETRIEVE NODE---")
    question = state["question"]
    if rag_retriever:
        documents = rag_retriever.invoke(question)
        # for doc in documents:
        #     print("Content:", doc.page_content)  # This is the actual chunk text
        #     print("Metadata:", doc.metadata)
        print(f"Retrieved {documents} documents for the question: '{question}'")
        return {"documents": documents, "question": question, "chat_history": state["chat_history"]}
    else:
        return {"documents": [], "question": question, "chat_history": state["chat_history"], "error": "Rag tool not initialized."}
    

# summ_prompt = langfuse.get_prompt("search-summarizer", {
#     "results": [
#         {"title": "News 1", "content": "India budget focus is healthcare..."},
#         {"title": "News 2", "content": "Education spending increased..."}
#     ]
# })
def generate(state: AgentState):
    """
    Generates a response using the LLM, incorporating retrieved documents.
    """
    print("---GENERATE NODE---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    # Create a prompt for the LLM
    # We include chat history and retrieved context
    context = "\n".join([doc.page_content for doc in documents])
    prompt = f"""
    You are a helpful AI assistant. Answer the user's question based on the provided context and chat history.
    Improve the quality of the answer by using the context provided and the chat history.
    If any calculations are needed, do them step by step and explain your reasoning.
    If the answer is not in the context, state that you don't know.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question: {question}
    Answer:
    """


    qa_prompt = langfuse.get_prompt(
    "qa-contextual",  # Prompt name
    {                # This second argument is a **single positional dict**
        "chat_history": str(chat_history),
        "context": context,
        "question": question
    }
)

   # Removed invalid langfuse.get_prompt call due to argument mismatch.
    print(f"Prompting LLM with: \n{prompt}...") # Print a snippet of the prompt
    response = llm.invoke(qa_prompt)
    return {"generation": response.content, "question": question, "documents": documents, "chat_history": chat_history}


# 4. Build a LangGraph graph

# Node: perform a web search via Tavily
def search_node(state: AgentState):
    
    resp = tavily_search.run({"query": state["question"]})

    # Normalize response
    if isinstance(resp, list):
        # wrap list into dict
        state["search_results"] = {"results": resp}
    elif isinstance(resp, dict) and "results" in resp:
        state["search_results"] = resp
    else:
        state["search_results"] = {"results": []}  # fallback

    return state

# Node: summarize results using LLM
def summarize_node(state: AgentState):
    prompt = (
        "Summarize the following results:\n"
        + "\n".join(
            [r.get("title", "") + ": " + r.get("content", "") 
             for r in state["search_results"]["results"]]
        )
    )
    summ_prompt = langfuse.get_prompt("search-summarizer", {
    "results": "\n".join(
        [f"{r['title']}: {r['content']}" for r in state["search_results"]["results"]]
    )
})

    summary_msg = llm.invoke(summ_prompt)
    summary_text = getattr(summary_msg, "content", str(summary_msg))
    
    # Return under "generation" too for consistency
    state["summary"] = {"content": summary_text}
    state["generation"] = summary_text
    return state

retrieval_tool = Tool(
    name="retrieval_tool",
    func=retrieve,
    description="Use for questions related to Budget Speech 2024-2025. Queries the local knowledge base."
)


search_tool = Tool(
    name="search_tool",
    func=search_node,
    description="Use for questions NOT related to Budget Speech 2024-2025. Performs a web search."
)

# tools = [search_tool, retrieval_tool]
# router_agent = create_react_agent(llm, tools, prompt)
# router_executor = AgentExecutor(agent=router_agent, tools=tools, verbose=True)
tools = [search_tool, retrieval_tool]
tool_names = ", ".join([t.name for t in tools])


def router(state: AgentState):
    """
    Uses the LLM to decide whether to use RAG (retrieve) or web search (search_node).
    Returns the name of the next node as a string.
    """
    question = state["question"]
    # Simple prompt to let LLM decide
    # prompt = f"""
    # Decide whether to answer the following question using the internal knowledge base (RAG) or a web search.
    # The knowledgebase only contains data about Budget Speech 2024-2025. It does not know anything else.
    # If the question is about Budget Speech 2024-2025, use the internal knowledge base (RAG).
    # Otherwise, use a web search.
    # Respond with only one word: 'retrieve' or 'search_node'.

    # Question: {question}
    # """
    # decision = llm.invoke(prompt)
    

    system_prompt = """You are a routing agent.
Decide which tool to use for a given question:
- If the question is about Budget Speech 2024-2025, call `retrieval_tool`.
- Otherwise, call `search_tool`.
You must always use one of these tools, never answer directly yourself.
"""

    prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("system", "Available tools:\n{tools}\n\nRemember: use only {tool_names}."),
    ("system", "Previous steps:\n{agent_scratchpad}")
])
    agent_scratchpad = ""   # start empty

    router_prompt = langfuse.get_prompt(
    "router-agent",
    {
        "input": question,
        "tools": str(tools),  # or json.dumps(tools)
        "tool_names": tool_names,
        "agent_scratchpad": agent_scratchpad
    }
)

    final_prompt = router_prompt.compile()

    router_agent = create_react_agent(llm, tools, final_prompt)
    router_executor = AgentExecutor(agent=router_agent, tools=tools, verbose=True)#, handle_parsing_errors=True)
    result = router_executor.invoke({"input": question})

    # # Extract string decision
    # decision_val = getattr(decision, "content", decision)
    # if isinstance(decision_val, list):
    #     decision_val = " ".join(str(item) for item in decision_val)

    # decision_str = str(decision_val).strip().lower()

    # Store decision in state so edges can use it
    state["decision"] = "search_node" if "search" in result else "retrieve"
    print(f"Router decision: {state['decision']} for question: '{question}'")
    #return {"question": state["question"], "decision": decision}

    return state



# Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("router", router)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("search_node", search_node)
workflow.add_node("summarize_node", summarize_node)

# Set the entry point
workflow.set_entry_point("router")
# ...existing code...

workflow.add_conditional_edges(
    "router",
    lambda state: state["decision"],
    {
        "retrieve": "retrieve",
        "search_node": "search_node"
    }
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("search_node", "summarize_node")
workflow.add_edge("generate", END)
workflow.add_edge("summarize_node", END)

# Compile the graph
app = workflow.compile()
print("LangGraph agent compiled.")

# --- 4. Chat Loop ---

print("\n--- Chatbot Ready! Type 'exit' to quit. ---")
chat_history = []
tokens_remaining = 10000  # Initialize total tokens for tracking
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Exiting chatbot. Goodbye!")
        break

    # Add user's message to chat history
    chat_history.append(HumanMessage(content=user_input))

    # Invoke the LangGraph agent
    try:
        inputs = {"question": user_input, "chat_history": chat_history}
        result = app.invoke(inputs)
        
        # Get the AI's response from the 'generation' field in the final state
        ai_response = result["generation"]
        print(f"AI Message: {ai_response}")

        # Update chat history with AI's response for the next turn
        chat_history.append(AIMessage(content=ai_response)) # LangChain's HumanMessage is used here for simplicity, but AIMessage is more appropriate for bot responses. For this example, it still works.

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your GOOGLE_API_KEY is correctly set.")
        chat_history.pop() # Remove the last user message if an error occurred to avoid polluting history
