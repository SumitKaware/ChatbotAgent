import os
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.tools import Tool
from model import tavily_search
# from langgraph.prebuilt import AgentState
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.render import render_text_description
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
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

    context = "\n".join([doc.page_content for doc in documents])

    # Get the prompt from Langfuse
    qa_prompt = langfuse.get_prompt(
        "qa-contextual",
        {
            "chat_history": str(chat_history),
            "context": context,
            "question": question
        }
    )

    print("Prompting LLM with 'qa-contextual' prompt from Langfuse...")
    chain = qa_prompt | llm | StrOutputParser()
    response = chain.invoke({
        "chat_history": str(chat_history),
        "context": context,
        "question": question
    })
    return {"generation": response, "question": question, "documents": documents, "chat_history": chat_history}


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
    print(f"Web search completed for query: '{state['question']}'")
    print(f"Search results: {json.dumps(state['search_results'], indent=2)}")
    return state

# Node: summarize results using LLM
def summarize_node(state: AgentState):
    # prompt = (
    #     "Summarize the following results:\n"
    #     + "\n".join(
    #         [r.get("title", "") + ": " + r.get("content", "") 
    #          for r in state["search_results"]["results"]]
    #     )
    # )

    # 1. Extract results from state
    results_list = [
        {"title": r["title"], "content": r["content"]}
        for r in state["search_results"]["results"]
    ]
    print("Results list for summarization:", results_list)
    # results_dict = {r["title"]: r["content"] for r in results_list}
    # print("Results dict values:", results_dict.values())
    # 2. Get Langfuse prompt template
    summ_prompt_template = langfuse.get_prompt("search-summarizer")

    # 3. Compile the prompt with variables
    summ_prompt = summ_prompt_template.compile(variables={"results": results_list})

    print(f"Prompting LLM with 'search-summarizer' prompt from Langfuse:\n{summ_prompt}")

    # 4. Convert results into dictionary (title -> content)
    #results_dict = {r["title"]: r["content"] for r in results_list}
    #print("Results dict:", results_dict.values())

    # 5. Run LLM using a chain or direct call (example with LangChain LLM)
    # chain = LLMChain(prompt=summ_prompt, llm=llm, output_parser=StrOutputParser())
    # summary = chain.run()
    # return summary

    #return results_dict  # For now, just returning the dict


    chain = results_dict | llm | StrOutputParser()
    summary_text = chain.invoke({
        "results": "\n".join(
            [f"{r['title']}: {r['content']}" for r in state["search_results"]["results"]]
        )
    })
    
    # Return under "generation" too for consistency
    state["summary"] = {"content": summary_text}
    state["generation"] = summary_text
    return state

def retrieval_decision(_):
    return "retrieve"

def search_decision(_):
    return "search_node"

retrieval_tool = Tool(
    name="retrieval_tool",
    func=retrieval_decision,
    description="Use for questions related to Budget Speech 2024-2025. Queries the local knowledge base."
)

search_tool = Tool(
    name="search_tool",
    func=search_decision,
    description="Use for questions NOT related to Budget Speech 2024-2025. Performs a web search."
)

def router(state: AgentState):
    """
    Uses the LLM to decide whether to use RAG (retrieve) or web search (search_node).
    """
    question = state["question"]
    tools = [search_tool, retrieval_tool]

    # Define the prompt directly in the code
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing agent. 
Decide whether to answer the question using the internal knowledge base (RAG) or a web search.

Rules:
- The knowledge base ONLY contains information about **Budget Speech 2024-2025** (funds, expansions, allocations, taxes, etc.).
- If the question is about Budget Speech 2024-2025 → use `retrieval_tool`.
- Otherwise → use `search_tool`.

Never wrap your output in code blocks or markdown.
Never answer the question yourself.
     Respond with only one word: 'retrieve' or 'search_node'.
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

    # Create the agent
    router_agent = create_tool_calling_agent(llm, tools, prompt)
    print("Router agent created with tools:", [tool.name for tool in tools])
    router_executor = AgentExecutor(agent=router_agent, tools=tools, verbose=True, handle_parsing_errors=True)
    print("Router executor created.")
    # Invoke the agent to get the decision
    result = router_executor.invoke({"input": question})
    print("Router agent result:", result)
    # The output of the executor is the return value of the tool function
    decision = result['output']
    state["decision"] = decision.strip().lower()
    print(f"Router decision: {decision} for question: '{question}'")
    
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
        chat_history.append(AIMessage(content=ai_response))

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your GOOGLE_API_KEY is correctly set.")
        chat_history.pop() # Remove the last user message if an error occurred to avoid polluting history
