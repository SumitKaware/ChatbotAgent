
import operator
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain.agents import AgentExecutor, create_react_agent

# 1. Define the Tools
# Search Engine Tool
search_tool = TavilySearchResults()

# RAG Tool
# A simple in-memory Chroma DB for demonstration
print("\n--- Demonstrating ChromaDB Loading and Retrieval ---")
current_script_dir = os.getcwd()
output_embedding_directory = os.path.join(current_script_dir, "embedded_content")
sample_chroma_path = os.path.join(output_embedding_directory, "chroma_db")
print(f"Loading ChromaDB from: {sample_chroma_path}")

rag_retriever = load_chroma_db_and_retriever(sample_chroma_path)
print("Retriever loaded:", rag_retriever)

rag_tool = Tool(
    name="retrieval_tool",
    description="Useful for answering questions about specific facts from the provided knowledge base.",
    func=lambda x: rag_retriever.invoke(x)
)

tools = [search_tool, rag_tool]

# 2. Define the Agent and Graph State
# This represents the state of the graph
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    agent_outcome: Annotated[Union[AgentAction, AgentFinish], operator.add]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

# 3. Define the Agent Logic
# The prompt for the agent
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. You have access to the following tools: {tools}"),
#     ("user", "Answer the following question: {input}"),
#     ("placeholder", "Begin your response by writing your thoughts, then the action you will take to answer the question, and finally the tool you will use. Available tools are: {tool_names}.\n\n{agent_scratchpad}"),
# ])

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You have access to the following tools: {tools}\n\n"
                   "Answer the user's questions as best you can."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(tool_names="search_tool, rag_tool")

# Create the agent runnable
agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Define the nodes of the graph
def run_agent(state: AgentState):
    agent_outcome = agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}

def execute_tool(state: AgentState):
    agent_action = state["agent_outcome"]
    tool_name = agent_action.tool
    tool_output = None
    
    # Execute the selected tool
    for tool in tools:
        if tool.name == tool_name:
            tool_output = tool.invoke(agent_action.tool_input)
            break
            
    return {"intermediate_steps": [(agent_action, str(tool_output))]}

# Define the conditional edges
def should_continue(state: AgentState):
    if isinstance(state["agent_outcome"], AgentFinish):
        return "end"
    else:
        return "continue"

# 4. Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", run_agent)
workflow.add_node("tool", execute_tool)

# Define the start and end points
workflow.set_entry_point("agent")

# Define edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tool", "end": END}
)
workflow.add_edge("tool", "agent")

# Compile the graph
app = workflow.compile()