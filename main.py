from logging import RootLogger

from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Literal
import uuid
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver #can use SQLLite and Postgress too
load_dotenv()

llm = init_chat_model(model="z-ai/glm-4.5-air:free", model_provider="openai") # using a free model from z-ai using openrouter

class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description="The intent of the message, one of 'chat', 'knowledge', or 'code'.")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None

def classify_intent(state: State) -> dict:
    structured_llm = llm.with_structured_output(IntentClassifier, method="json_mode")
    result = structured_llm.invoke([
        {"role": "system", "content": "You are an intent classifier that determines the intent of the user's message if the user wants to chat ('chat'), retrieve knowledge ('knowledge') or change code ('code'). Respond with a JSON object containing a 'message_intent' key matching the schema."},
        {"role": "user", "content": state["messages"][-1].content}
    ])
    return {"message_intent": result.message_intent}

def prompt_llm_chat(state: State) -> dict:
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages":{'role': 'assistant', 'content': response.content}}

def prompt_llm_rag(state: State) -> dict:
    messages = [{'role': 'system', 'content': 'no matter what the user says, always respond with "i am the RAG agent"'}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages":{'role': 'assistant', 'content': response.content}}

def prompt_llm_code(state: State) -> dict:
    messages = [{'role': 'system', 'content': 'no matter what the user says, always respond with "i am the code agent"'}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages":{'role': 'assistant', 'content': response.content}}

graph_builder = StateGraph(State)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("prompt_llm_chat", prompt_llm_chat)
graph_builder.add_node("prompt_llm_rag", prompt_llm_rag)
graph_builder.add_node("prompt_llm_code", prompt_llm_code)

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent",lambda state: state["message_intent"], {'chat':'prompt_llm_chat', 'knowledge': 'prompt_llm_rag', 'code': 'prompt_llm_code'})

graph_builder.add_edge("prompt_llm_chat", END)
graph_builder.add_edge("prompt_llm_rag", END)
graph_builder.add_edge("prompt_llm_code", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

config = {'configurable': {'thread_id': str(uuid.uuid4())}}

while True: 
    user_input = input("User: ")
    result = graph.invoke({'messages':[{'role':'user', 'content': user_input}]}, config)
    print("Bot:", result['messages'][-1].content)

    

