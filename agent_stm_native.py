import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from operator import add
from datetime import datetime
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
try:
    chat_model = ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            task="text-generation",
            huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
            max_new_tokens=256,
        )
    )
except Exception as e:
    print(f"Error initializing Hugging Face LLM: {e}")
    exit()

checkpointer = InMemorySaver()#fresh instance on each run
print("new checkpointer created→")
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add] 

def chat_node(state: ChatState) -> dict:
    response = chat_model.invoke(state["messages"]) 
    return {"messages": [response]}

graph_builder=StateGraph(ChatState)
graph_builder.add_node("chatbot", chat_node)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot") 

app=graph_builder.compile(checkpointer=checkpointer)

def run_conversation(user_message: str):
    config = {"configurable": {"thread_id": "demo_session_001"}}
    
    print(f"user message: {user_message}")
    response = app.invoke(
        {"messages": [HumanMessage(content=user_message)]}, 
        config=config
    )
    ai_response = response['messages'][-1].content.strip()
    print(f"ai agent response: {ai_response}\n")
    return ai_response

if __name__ == "__main__":
    run_conversation("Hi, my name is Jash Thakkar and I am a developer.")#1st convo - name input
    response = run_conversation("What is my name?")#2nd convo - name recall
    
    if "jash" in response.lower() or "thakkar" in response.lower():
        print("NAME RECALLED")
    else:
        print("NAME NOT RECALLED")