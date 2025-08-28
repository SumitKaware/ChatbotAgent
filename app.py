import os
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
# from langgraph.prebuilt import AgentState
from langchain.agents import AgentExecutor, create_react_agent
from model import llm  # Importing the LLM and embeddings from model.py
from pdfloader import vector_db
from storing_embeddings import load_chroma_db_and_retriever  # Importing raw documents from pdfloader.py
#
# Create a vector database

#vector_db()  # Initialize the vector store with a sample query

search_tool = TavilySearchResults()
# --- Demonstrate loading ChromaDB and using retriever ---
print("\n--- Demonstrating ChromaDB Loading and Retrieval ---")
current_script_dir = os.getcwd()
output_embedding_directory = os.path.join(current_script_dir, "embedded_content")
sample_chroma_path = os.path.join(output_embedding_directory, "chroma_db")
print(f"Loading ChromaDB from: {sample_chroma_path}")

rag_retriever = load_chroma_db_and_retriever(sample_chroma_path)
print("Retriever loaded:", rag_retriever)
# query = "What is the profit for Q4?"
# print(f"\nQuerying ChromaDB: '{query}'")
# try:
#     retrieved_docs = retriever.invoke(query)
# except Exception as e:
#     print(f"Error querying ChromaDB: {e}")
#     retrieved_docs = []
# if retriever:
#     query = "What is the amount allocated for education?"
#     print(f"\nQuerying ChromaDB: '{query}'")
    
#     retrieved_docs = retriever.invoke(query)

#     print("\n--- Retrieved Documents ---")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"Document {i+1} (Type: {doc.metadata.get('type')}, Page: {doc.metadata.get('page_num')}):")
#         print(f"  Content: {doc.page_content[:150]}...") # Print first 150 chars of content
#         print(f"  Metadata: {doc.metadata}\n")
# else:
#     print("Could not load ChromaDB for demonstration.")



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


# Define the nodes (functions) in our graph
rag_tool = Tool(
    name="retrieval_tool",
    description="Useful for answering questions about specific facts from the provided knowledge base.",
    func=lambda x: rag_retriever.invoke(x)
)

tools = [search_tool, rag_tool]

def retrieve(state: AgentState):
    """
    Retrieves documents based on the user's question.
    """
    print("---RETRIEVE NODE---")
    question = state["question"]
    if rag_tool:
        documents = rag_tool.func(question)
        # for doc in documents:
        #     print("Content:", doc.page_content)  # This is the actual chunk text
        #     print("Metadata:", doc.metadata)
        print(f"Retrieved {documents} documents for the question: '{question}'")
        return {"documents": documents, "question": question, "chat_history": state["chat_history"]}
    else:
        return {"documents": [], "question": question, "chat_history": state["chat_history"], "error": "Rag tool not initialized."}
    

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
    print(f"Prompting LLM with: \n{prompt}...") # Print a snippet of the prompt
    response = llm.invoke(prompt)
    return {"generation": response.content, "question": question, "documents": documents, "chat_history": chat_history}


tavily_tool = TavilySearchResults(
    max_results=3,
    topic="general",
    include_answer=True,
    include_raw_content=False,
    search_depth="basic",
)

# 4. Build a LangGraph graph

# Node: perform a web search via Tavily
def search_node(state: AgentState):
    resp = tavily_tool.run({"query": state["question"]})
    state["search_results"] = resp
    return state

# Node: summarize results using LLM
def summarize_node(state: AgentState):
    prompt = (
        "Summarize the following results:\n"
        + "\n".join([r["title"] + ": " + r.get("content", "") for r in state["search_results"]["results"]])
    )
    summary_msg = llm.invoke(prompt)
    # If summary_msg is a BaseMessage, extract its content
    state["summary"] = {"content": summary_msg.content} if hasattr(summary_msg, "content") else {"content": summary_msg}
    return state
    # summary = llm.invoke(prompt)
    # state["summary"] = summary
    return state


# Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes to the workflow
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

# Set the entry point
workflow.set_entry_point("retrieve")

# Define the edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

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
        # The agent expects the initial state.
        # We pass the current chat_history to ensure it's propagated.
        # ...existing code...
        #inputs = AgentState({"question": user_input, "chat_history": chat_history})
#         inputs = AgentState(
#     question=user_input,
#     chat_history=chat_history,
#     documents=[],
#     generation=""
# )
#         result = app.invoke(inputs)
# ...existing code...
        inputs = {"question": user_input, "chat_history": chat_history}
        result = app.invoke(inputs)
        # inputs = AgentState.from_dict({"question": user_input, "chat_history": chat_history})
        # result = app.invoke(inputs)
        # print(len(str(inputs)), len(str(result)))
        # input_tokens = len(str(inputs))/4
        # output_tokens = len(str(result))/4
        # print(f"Input: {inputs}, \n\nResult: {result}")
        # # print(f"Input tokens: len({input_tokens}), Output tokens: len({output_tokens})")
        # # Calculate total tokens used and remaining

        # total_tokens_used = input_tokens + output_tokens
        # tokens_remaining = tokens_remaining - total_tokens_used
        # print(f"Tokens used: {total_tokens_used}, Tokens remaining: {tokens_remaining}")

        # Get the AI's response from the 'generation' field in the final state
        ai_response = result["generation"]
        print(f"AI Message: {ai_response}")

        # Update chat history with AI's response for the next turn
        chat_history.append(AIMessage(content=ai_response)) # LangChain's HumanMessage is used here for simplicity, but AIMessage is more appropriate for bot responses. For this example, it still works.

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your GOOGLE_API_KEY is correctly set.")
        chat_history.pop() # Remove the last user message if an error occurred to avoid polluting history
