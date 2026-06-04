from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, END, START

load_dotenv()

llm = init_chat_model(model="z-ai/glm-4.5-air:free", model_provider="openai") # using a free model from z-ai using openrouter
def prompt_llm(state: MessagesState) -> dict:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("prompt_llm", prompt_llm) # langgraph graph always starts with START and ends with END
graph_builder.add_edge(START, 'prompt_llm')
graph_builder.add_edge('prompt_llm', END)

graph = graph_builder.compile()

user_input = input("Enter your message: ")
result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
print(result)
