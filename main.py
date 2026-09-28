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

# answers accepted at the HIL prompt. the two sets must never share a word, and anything that is
# in neither set is treated as a modification of the coding request
APPROVE_WORDS = {'y', 'yes', 'approve', 'ok', 'okay', 'go', 'run', 'proceed', 'accept'}
DENY_WORDS = {'n', 'no', 'disapprove', 'deny', 'disallow', 'decline', 'stop', 'wait', 'dont run', 'exit', 'e'}

CODE_CONTEXT_MESSAGES = 6      # how many recent messages opencode gets as conversation context
CODE_CONTEXT_MAX_CHARS = 2000  # each of those messages is cut to this length (old opencode output can be huge)
CODE_TIMEOUT_SECONDS = 600     # opencode is stopped if it runs longer than this

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None
    code_request: str | None  # the coding request waiting for approval: the original ask plus any revisions

    next_node: str | None

def classify_intent(state: State) -> dict:
    last_message = state["messages"][-1].content
    structured_llm = llm.with_structured_output(IntentClassifier, method="json_mode")
    result = structured_llm.invoke([
        {"role": "system", "content": "You are an intent classifier that determines the intent of the user's message if the user wants to chat ('chat'), retrieve knowledge ('knowledge') or change code ('code'). Respond with a JSON object containing a 'message_intent' key matching the schema."},
        {"role": "user", "content": last_message}
    ])
    # every turn starts here, so the pending coding request is reset on each turn.
    # this way a request from an earlier turn can never leak into this one
    code_request = last_message if result.message_intent == 'code' else None
    return {"message_intent": result.message_intent, "code_request": code_request}

def accept_coding(state: State) -> dict:
    # NOTE: on resume langgraph re-runs this node from the top, so nothing before interrupt() may have side effects
    request = state["code_request"]
    decision = interrupt(f"we are about to run opencode with request:\n\n{request}\n\ntype yes to accept, no to deny, or type a modification")

    raw = str(decision).strip()
    text = raw.lower()  # lowercase only for matching the approve/deny words

    if text in APPROVE_WORDS:
        return {'next_node': 'prompt_llm_code'}

    if text in DENY_WORDS:
        return {"messages": [{'role': 'assistant', 'content': 'Coding request denied by user.'}], 'next_node': 'denied'}

    if not raw:
        # empty answer: ask again without changing anything
        return {'next_node': 'accept_coding'}

    # anything else is a modification. the original request is kept and the revision is appended
    # (with its original casing, since file names and code are case sensitive). the revision is
    # saved as a user message because the user wrote it
    return {
        "messages": [{'role': 'user', 'content': raw}],
        'code_request': f"{request}\n\nRevision from the user: {raw}",
        'next_node': 'accept_coding',
    }
    
    
def prompt_llm_chat(state: State) -> dict:
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages":{'role': 'assistant', 'content': response.content}}

def prompt_llm_rag(state: State) -> dict:
    query = state['messages'][-1].content 
    documents = vector_store.similarity_search(query, k = 3)
    context = "\n\n".join([doc.page_content for doc in documents])
    
    messages = [{'role': 'system', 'content': f"you are the RAG AGENT. answer the user using only the context below. if the answer is not in it, say you dont know. \n\nContext:\n{context}"}] + state["messages"]
    response = llm.invoke(messages)
    return {"messages":{'role': 'assistant', 'content': response.content}}

def format_recent_history(messages: list) -> str:
    """Turn the last few messages into plain text so opencode knows what the conversation was about."""
    lines = []
    for message in messages[-CODE_CONTEXT_MESSAGES:]:
        role = {'human': 'user', 'ai': 'assistant'}.get(message.type, message.type)
        content = str(message.content)
        if len(content) > CODE_CONTEXT_MAX_CHARS:
            content = content[:CODE_CONTEXT_MAX_CHARS] + " ...[truncated]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def prompt_llm_code(state: State) -> dict:
    # opencode is a separate process and can't see the graph's memory, so the recent conversation
    # is written into the prompt together with the approved request
    code_prompt = (
        "Recent conversation with the user (for context only):\n"
        f"{format_recent_history(state['messages'])}\n\n"
        "Task approved by the user (do this):\n"
        f"{state['code_request']}"
    )
    workspace = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workspace")
    
    # Option A: Using OpenCode (Default)
    opencode_path = os.path.expanduser('~/.opencode/bin/opencode')
    command = [opencode_path, 'run', code_prompt, '--dangerously-skip-permissions']

    # Option B: Using Claude Code (Alternative)
    # To use Claude Code instead, make sure it is installed globally and uncomment this:
    # command = ['claude', '-p', code_prompt, '--permission-mode', 'acceptEdits']

    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            capture_output=True,
            timeout=CODE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return {"messages":{'role': 'assistant', 'content': f"could not start the coding agent: {command[0]} was not found. install it or switch to the other option in prompt_llm_code."}}
    except subprocess.TimeoutExpired:
        return {"messages":{'role': 'assistant', 'content': f"the coding agent did not finish within {CODE_TIMEOUT_SECONDS} seconds and was stopped."}}

    output = result.stdout.strip() or result.stderr.strip() or "(the coding agent produced no output)"
    if result.returncode != 0:
        output = f"the coding agent exited with code {result.returncode}:\n{output}"
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

def main():
    # drawing the png calls the mermaid.ink web api, so it fails offline. that shouldn't stop the agent
    try:
        graph.get_graph().draw_mermaid_png(output_file_path='agent_graph_after_looping_with_HIL.png')
    except Exception as error:
        print(f"(skipped drawing the graph png: {error})")

    config = {'configurable': {'thread_id': str(uuid.uuid4())}}

    while True:
        try:
            user_input = input("User: ")
            if not user_input.strip():
                continue

            result = graph.invoke({'messages':[{'role':'user', 'content': user_input}]}, config)

            while '__interrupt__' in result:
                prompt = result['__interrupt__'][0].value
                decision = input(f'{prompt}\n> ')
                result = graph.invoke(Command(resume=decision), config)

            print("Bot:", result['messages'][-1].content)
        except (KeyboardInterrupt, EOFError):
            # ctrl+c / ctrl+d quits cleanly
            print()
            break
        except Exception as error:
            # one failed turn (bad json from the classifier, rate limit, network error...) shouldn't
            # kill the session. the checkpointer still holds the conversation, so just keep going
            print(f"Bot: something went wrong ({type(error).__name__}: {error}). you can keep chatting.")

if __name__ == "__main__":
    main()
