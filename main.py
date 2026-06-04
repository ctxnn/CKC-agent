from logging import RootLogger

from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Literal
import uuid
import subprocess
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document 
# NOTE: If you have paid OpenAI/OpenRouter credits, you can uncomment these:
# from langchain_core.vectorstores import InMemoryVectorStore 
# from langchain_openai import OpenAIEmbeddings

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver #can use SQLLite and Postgress too
from langgraph.types import interrupt, Command 
load_dotenv()

llm = init_chat_model(model="z-ai/glm-4.5-air:free", model_provider="openai") # using a free model from z-ai using openrouter

KNOWLEDGE = ["James Harris Simons (April 25, 1938 – May 10, 2024) was an American hedge fund manager, investor, mathematician, and philanthropist.[4] At the time of his death, Simons's net worth was estimated to be $31.4 billion, making him the 55th-richest person in the world.[4] He was the founder of Renaissance Technologies, a quantitative hedge fund based in East Setauket, New York. He and his fund are known to be quantitative investors, using mathematical models and algorithms to make investment gains from market inefficiencies. Due to the long-term aggregate investment returns of Renaissance and its Medallion Fund, Simons was called the greatest investor on Wall Street and more specifically the most successful hedge fund manager of all time", "Simons developed the Chern–Simons form (with Shiing-Shen Chern), and contributed to the development of string theory by providing a theoretical framework to combine geometry and topology with quantum field theory.[8]", "In 1994, Simons and his wife, Marilyn, founded the Simons Foundation to support research in mathematics and fundamental sciences. The foundation is the top benefactor of Stony Brook University, Marilyn's alma mater, and is a major contributor to his alma maters, the Massachusetts Institute of Technology and the University of California, Berkeley. Simons was a member of the boards of the Stony Brook Foundation, the MIT Corporation, and the Simons Laufer Mathematical Sciences Institute in Berkeley, and chaired the boards of Math for America, the Simons Foundation, and Renaissance Technologies.[9][10] In 2023, the Simons Foundation gave $500 million to Stony Brook University, the second-largest donation to a public university in U.S. history.[11] In 2016, the International Astronomical Union named asteroid 6618 Jimsimons, which Clyde Tombaugh discovered in 1936, after Simons in honor of his contributions to mathematics and philanthropy.[12] "]

# NOTE: If you have paid OpenAI/OpenRouter credits, you can uncomment this block:
# vector_store = InMemoryVectorStore(OpenAIEmbeddings(model='text-embedding-3-small'))
# vector_store.add_documents([Document(page_content=text) for text in KNOWLEDGE])

class SimpleLocalSearch:
    """A lightweight, local keyword-similarity search that doesn't need API keys or heavy packages."""
    def __init__(self, documents: list[Document]):
        # Store the knowledge documents locally
        self.documents = documents

    def similarity_search(self, query: str, k: int = 3) -> list[Document]:
        # Split the query into a set of lowercased, unique words
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            doc_words = doc.page_content.lower().split()
            # Score the document by counting how many query words it contains (simple overlap)
            overlap = sum(1 for word in query_words if word in doc_words)
            scored_docs.append((overlap, doc))
        # Sort the documents by their overlap score in descending order (highest matches first)
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        # Return the top k matching documents
        return [doc for score, doc in scored_docs[:k]]

# Initialize the local search tool with the knowledge document corpus
vector_store = SimpleLocalSearch([Document(page_content=text) for text in KNOWLEDGE])

class IntentClassifier(BaseModel):
    message_intent: Literal['chat', 'knowledge', 'code'] = Field(..., description="The intent of the message, one of 'chat', 'knowledge', or 'code'.")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None

    next_node: str | None

def classify_intent(state: State) -> dict:
    structured_llm = llm.with_structured_output(IntentClassifier, method="json_mode")
    result = structured_llm.invoke([
        {"role": "system", "content": "You are an intent classifier that determines the intent of the user's message if the user wants to chat ('chat'), retrieve knowledge ('knowledge') or change code ('code'). Respond with a JSON object containing a 'message_intent' key matching the schema."},
        {"role": "user", "content": state["messages"][-1].content}
    ])
    return {"message_intent": result.message_intent}

def accept_coding(state: State) -> dict:
    user_prompt = state["messages"][-1].content
    decision = interrupt(f"we areabout to run opencode with request: {user_prompt}\n\naccept the above request to continue? \n\ntype yes, no or a modification")

    text = str(decision).strip().lower() 

    if text in ["yes","no", 'approve', 'ok', 'okay', 'go', 'run', 'proceed', 'accept', 'y', 'yes', 'ok', 'okay', 'go', 'run', 'proceed', 'accept']:
        return {'next_node': 'prompt_llm_code'}
    
    if text in ['n', 'no', 'disapprove', 'deny', 'disallow', 'decline', 'stop', 'wait', 'dont run', 'exit', 'e', 'no', 'disapprove', 'deny', 'disallow', 'decline', 'stop', 'wait', 'dont run', 'exit']:
        return {"messages": [{'role': 'assistant', 'content': 'Coding request denied by user.'}], 'next_node': 'denied'}

    return {"messages": [{'role': 'assistant', 'content': text}], 'next_node': 'accept_coding'}
    
    
def prompt_llm_chat(state: State) -> dict:
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages":{'role': 'assistant', 'content': response.content}}

def prompt_llm_rag(state: State) -> dict:
    query = state['messages'][-1].content 
    documents = vector_store.similarity_search(query, k = 3)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    messages = [{'role': 'system', 'content': f"you are the RAG AGENT. answer the user using only the context below. if the answer is not in it, say you dont know. \n\nContext:\n{context}\n\nMessages"}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages":{'role': 'assistant', 'content': response.content}}

def prompt_llm_code(state: State) -> dict:
    user_prompt = state['messages'][-1].content 
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
    
    # Option A: Using OpenCode (Default)
    opencode_path = os.path.expanduser('~/.opencode/bin/opencode')
    result = subprocess.run(
        [opencode_path, 'run', user_prompt, '--dangerously-skip-permissions'], 
        cwd=workspace,
        text=True,
        capture_output=True,
    )
    
    # Option B: Using Claude Code (Alternative)
    # To use Claude Code instead, make sure it is installed globally and uncomment this:
    # result = subprocess.run(
    #     ['claude', '-p', user_prompt, '--permission-mode', 'accept-edits'], 
    #     cwd=workspace,
    #     text=True,
    #     capture_output=True,
    # )
    
    output = result.stdout.strip() or result.stderr.strip()
    return {"messages":{'role': 'assistant', 'content': output}}

graph_builder = StateGraph(State)

graph_builder.add_node("classify_intent", classify_intent)
graph_builder.add_node("prompt_llm_chat", prompt_llm_chat)
graph_builder.add_node("prompt_llm_rag", prompt_llm_rag)
graph_builder.add_node("prompt_llm_code", prompt_llm_code)
graph_builder.add_node('accept_coding', accept_coding)

graph_builder.add_conditional_edges('accept_coding', lambda state: 'end' if state.get('next_node') == 'denied' else state['next_node'], {'end': END, 'prompt_llm_code': 'prompt_llm_code', 'accept_coding':'accept_coding'})

graph_builder.add_edge(START, "classify_intent")
graph_builder.add_conditional_edges("classify_intent",lambda state: state["message_intent"], {'chat':'prompt_llm_chat', 'knowledge': 'prompt_llm_rag', 'code': 'accept_coding'})

graph_builder.add_edge("prompt_llm_chat", END)
graph_builder.add_edge("prompt_llm_rag", END)
graph_builder.add_edge("prompt_llm_code", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

graph.get_graph().draw_mermaid_png(output_file_path='agent_graph_after_looping_with_HIL.png')

config = {'configurable': {'thread_id': str(uuid.uuid4())}}

while True: 
    user_input = input("User: ")
    result = graph.invoke({'messages':[{'role':'user', 'content': user_input}]}, config)

    while '__interrupt__' in result: 
        prompt = result['__interrupt__'][0].value
        decision = input(f'{prompt}\n\nContinue? Y/N: ')
        result = graph.invoke(Command(resume=decision), config)
        
    print("Bot:", result['messages'][-1].content)

    

