# Explaining `main.py`: Interactive Agent with HIL Loop and RAG

This document provides a detailed walkthrough of the agentic graph architecture implemented in [main.py](main.py). 

---

## 1. Graph Architecture Diagram

Below is the graph representing the state transitions of the agent system. The diagram is saved in the repo root as `agent_graph_after_looping_with_HIL.png` and represents the flow:

![Agent Graph](agent_graph_after_looping_with_HIL.png)

---

## 2. Key Components Explained

### A. Graph State Definition (`State`)
The state keeps track of the conversation and routing information across graph steps:
```python
class State(TypedDict):
    messages: Annotated[list, add_messages] # Append-only list of messages (conversation history)
    message_intent: str | None              # Intent classified by classify_intent node
    code_request: str | None                # Coding request waiting for approval (original ask + revisions)
    next_node: str | None                   # Destination node for HIL routing
```

* **`messages`**: Uses the `add_messages` reducer to automatically append new user inputs and assistant responses to the conversation memory.
* **`code_request`**: The exact text that will be sent to OpenCode as the task. It starts as the user's coding message and grows with every revision typed at the approval prompt. Keeping it in its own field (instead of reading `messages[-1]`) is what lets a revision *add to* the request instead of replacing it.
* **`next_node`**: Stores the target node determined by the Human-In-The-Loop (HIL) check to govern conditional routing.

> Every field is saved by the checkpointer between turns, so values from an earlier turn are still there on the next one. That's why `classify_intent` resets `code_request` on every turn (see below).

---

### B. Intent Classification (`classify_intent`)
This node acts as the router for incoming user messages:
* **Logic:** Uses the free LLM `z-ai/glm-4.5-air` with `with_structured_output` configured in `json_mode`. It only looks at the latest message.
* **Output:** Returns `message_intent` (one of `'chat'`, `'knowledge'`, or `'code'`, validated by the `IntentClassifier` schema) and `code_request`:
  * intent `'code'` → `code_request` = the user's message
  * anything else → `code_request` = `None`

  Because every turn passes through this node, a coding request from an earlier turn can never leak into a later one.
* **Conditional Routing:**
  * `'chat'` $\rightarrow$ `prompt_llm_chat`
  * `'knowledge'` $\rightarrow$ `prompt_llm_rag`
  * `'code'` $\rightarrow$ `accept_coding` (Human-in-the-loop validation)

---

### C. Human-in-the-Loop Node (`accept_coding`)
When the user asks to change code, the graph routes here first to obtain explicit authorization before executing local commands:

```python
APPROVE_WORDS = {'y', 'yes', 'approve', 'ok', 'okay', 'go', 'run', 'proceed', 'accept'}
DENY_WORDS = {'n', 'no', 'disapprove', 'deny', 'disallow', 'decline', 'stop', 'wait', 'dont run', 'exit', 'e'}

def accept_coding(state: State) -> dict:
    request = state["code_request"]
    # Triggers a LangGraph interrupt, pausing graph execution until the user answers
    decision = interrupt(f"we are about to run opencode with request:\n\n{request}\n\ntype yes to accept, no to deny, or type a modification")

    raw = str(decision).strip()
    text = raw.lower()  # lowercase only for matching the approve/deny words

    if text in APPROVE_WORDS:
        return {'next_node': 'prompt_llm_code'}

    if text in DENY_WORDS:
        return {"messages": [{'role': 'assistant', 'content': 'Coding request denied by user.'}], 'next_node': 'denied'}

    if not raw:
        return {'next_node': 'accept_coding'}   # empty answer: ask again

    # modification: keep the original request, append the revision, loop back for approval
    return {
        "messages": [{'role': 'user', 'content': raw}],
        'code_request': f"{request}\n\nRevision from the user: {raw}",
        'next_node': 'accept_coding',
    }
```

* **`interrupt()`**: Pauses the graph. In the outer loop, the script catches the interrupt, asks the user for input, and uses `Command(resume=decision)` to resume execution.
* **Resume re-runs the node from the top.** On resume LangGraph starts `accept_coding` again from its first line, and this time `interrupt()` returns the user's answer instead of pausing. So nothing before `interrupt()` may have side effects (here it only reads state).
* **Approve / deny words** live in two module-level sets that must never share a word. Answers are compared case-insensitively and must match a whole word from a set (`"yes please"` is a modification, not an approval).
* **Revision Loop:** Any other non-empty answer is a modification. The routing edge sends the graph back to `accept_coding`, which is a *new* run of the node, so it pauses again and shows the combined request for approval. This repeats until the user approves or denies. Details:
  * The revision is **appended** to `code_request`, so the original ask is kept.
  * The revision keeps its **original casing**, because file names and code are case sensitive.
  * The revision is saved in `messages` as a **user** message, because the user wrote it.

---

### D. Knowledge Retrieval Node (`prompt_llm_rag`)
Because calling standard embedding models can be expensive or trigger API payment issues, this node uses a **custom local search**:

1. **`SimpleLocalSearch`**: Scores your preloaded `KNOWLEDGE` paragraphs by checking word-overlap with the user's query.
2. **Context Assembly**: Concatenates the top 3 highest-matching paragraphs.
3. **Prompt Grounding**: Assembles a system prompt forcing the LLM to only answer using the retrieved context (or say *"I do not know"* if missing), followed by the full conversation history.
4. **Known limitation:** the corpus has exactly 3 paragraphs and `k=3`, so every query currently gets all 3 back. The ranking only starts to matter once `KNOWLEDGE` grows. The word matching is also naive: `"simons?"` doesn't match `simons`, and common words like `the` count as matches.

---

### E. Code Execution Node (`prompt_llm_code`)
Once a coding request is approved, this node runs a coding agent inside the `workspace` subdirectory.

**1. Building the prompt.** OpenCode runs as a separate process and can't see the graph's memory. So the node writes the recent conversation into the prompt, followed by the approved request:
```text
Recent conversation with the user (for context only):
user: can you add hello world to the readme file?
user: write it in CAPS

Task approved by the user (do this):
can you add hello world to the readme file?

Revision from the user: write it in CAPS
```
`format_recent_history` produces the first part:
* It takes the last `CODE_CONTEXT_MESSAGES` (6) messages.
* It labels each one `user` / `assistant`.
* It cuts each message to `CODE_CONTEXT_MAX_CHARS` (2000) characters, so a huge earlier OpenCode output doesn't flood the prompt.

**2. Running the agent.**
* **OpenCode (Default):**
  ```bash
  ~/.opencode/bin/opencode run "<prompt>" --dangerously-skip-permissions
  ```
  The `--dangerously-skip-permissions` flag is used because the request was already approved by you in `accept_coding`. It still lets OpenCode do anything your user account can, so only approve requests you have actually read.
* **Claude Code (Alternative):** Swap the `command` line for the commented one:
  ```bash
  claude -p "<prompt>" --permission-mode acceptEdits
  ```

**3. Handling failures.** Every outcome is returned to the user as the bot's reply instead of crashing the program:

| Situation | Reply |
|---|---|
| binary not installed (`FileNotFoundError`) | `could not start the coding agent: ... was not found` |
| runs longer than `CODE_TIMEOUT_SECONDS` (600) | `the coding agent did not finish within 600 seconds and was stopped.` |
| non-zero exit code | `the coding agent exited with code N:` + its output |
| success | its stdout (or stderr if stdout is empty) |

---

## 3. The Execution Loop (`main()`)

The runtime loop controls how messages flow in and out of the graph and handles interrupts. It lives in `main()` behind `if __name__ == "__main__":`, so `main.py` can be imported (e.g. to test the graph with a fake LLM) without starting the chat.

```python
def main():
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

            # Run graph with new user message
            result = graph.invoke({'messages':[{'role':'user', 'content': user_input}]}, config)

            # If the graph was paused by a node calling `interrupt()`
            while '__interrupt__' in result:
                prompt = result['__interrupt__'][0].value
                decision = input(f'{prompt}\n> ')
                # Resume the paused graph execution with the user's decision
                result = graph.invoke(Command(resume=decision), config)

            print("Bot:", result['messages'][-1].content)
        except (KeyboardInterrupt, EOFError):
            print()
            break
        except Exception as error:
            print(f"Bot: something went wrong ({type(error).__name__}: {error}). you can keep chatting.")
```

1. **Graph drawing**: `draw_mermaid_png` calls the mermaid.ink web API. It's wrapped in `try` so that being offline doesn't stop the agent from starting.
2. **`graph.invoke`**: Runs the graph with the new user message.
3. **`__interrupt__` Check**: If the graph is paused by `accept_coding`, the key `__interrupt__` is present in the result. The inner `while` keeps asking until the graph finishes, because a modification pauses the graph again.
4. **`Command(resume=decision)`**: Feeds the user's answer back into the graph, which resumes at the node that paused.
5. **Error handling**:
   * `Ctrl+C` / `Ctrl+D` quit cleanly.
   * Any other exception (bad classifier JSON, rate limit, network error) is printed, and the loop continues. The conversation is still in the checkpointer under the same `thread_id`, and the next message simply starts a new run from `START`.
   * The message that failed does stay in the history without a reply.

---

## 4. Imports & Libraries Reference

Here is a breakdown of all the libraries and packages imported in `main.py`:

### Standard Python Libraries
* **`os`**: Standard operating system interface library. Used to resolve workspace folders (`os.path.join`) and expand home directory paths (`os.path.expanduser`).
* **`uuid`**: Universal Unique Identifier generator. Used to create a random `thread_id` (`uuid.uuid4()`) for keeping conversation sessions separate in the checkpointer.
* **`subprocess`**: Python's system runner. Spawns child processes to invoke CLI agents like `opencode` or `claude code` headlessly and capture their output (`capture_output=True`).
* **`typing (TypedDict, Annotated, Literal)`**:
  * `TypedDict`: Defines the schema of the LangGraph state.
  * `Annotated`: Used to attach custom metadata and reducer functions to state variables (like `add_messages` to the `messages` list).
  * `Literal`: Restricts intent types strictly to `'chat'`, `'knowledge'`, or `'code'` to prevent invalid classifications.

### Pydantic (Data Validation)
* **`BaseModel`**: The base class for defining schemas in Python. Used to declare the `IntentClassifier` structure.
* **`Field`**: Attaches metadata, default values, and semantic descriptions to schema properties, giving context to the LLM when outputting JSON.

### LangChain & OpenAI Integration
* **`load_dotenv` (`python-dotenv`)**: Parses and loads environment variables from a local `.env` file into python's runtime, providing access to `OPENAI_API_KEY` and `OPENAI_API_BASE`.
* **`init_chat_model`**: A helper function from LangChain that initializes different chat models dynamically via unified parameters.
* **`Document`**: The standard text container class inside LangChain used to store page contents and metadata, consumed by retrieval tools.

### LangGraph (State Machine Agent Framework)
* **`StateGraph`**: The constructor class to register nodes, edges, state transitions, and compile the final runnable agent.
* **`START` & `END`**: Graph boundary sentinels indicating where execution begins and ends.
* **`add_messages`**: An append-reducer function. Instead of replacing the `messages` field, it merges new messages into the existing conversation list.
* **`InMemorySaver`**: A basic checkpointer that stores graph state in RAM. This provides the agent with "short-term memory" during chat loop iterations under the same `thread_id`.
* **`interrupt`**: Halts execution at a node (useful for Human-in-the-Loop validation) and pauses until input is received.
* **`Command`**: Used outside the graph to resume from an `interrupt` boundary, feeding the user's input/decision back to the graph.

