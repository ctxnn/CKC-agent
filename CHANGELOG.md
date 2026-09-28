# Changelog

## Fixes to the HIL loop, OpenCode context and error handling

These fixes came out of a review of the first working version of `main.py` (commit `486c43a`). The graph's shape (nodes and edges) is unchanged, so `agent_graph_after_looping_with_HIL.png` still matches. What changed is what the nodes do.

### Bug fixes

**1. Typing `no` at the approval prompt ran OpenCode.**
* **Before:** the approve list was `["yes", "no", 'approve', ...]`, and it was checked before the deny list. So the most natural way to refuse (`no`) ran OpenCode with `--dangerously-skip-permissions`. (`n`, `deny`, `stop` did deny correctly.)
* **Now:** approve and deny words are two separate sets, `APPROVE_WORDS` and `DENY_WORDS`, with no word in both. The duplicate entries in the old lists are gone too.

**2. A modification replaced the original request, lowercased it and pretended to be the assistant.**
* **Before:** the answer was lowercased and appended as an **assistant** message. The next round then treated `messages[-1]` as the whole request. As a result:
  * the original ask was lost,
  * file names changed case (`README.md` → `readme.md`),
  * the chat history had a fake assistant turn in it.
* **Now:** a new `code_request` state field holds the pending request. Each revision is appended to it as `Revision from the user: ...` with its original casing, and it's also saved to history as a **user** message. Lowercasing is only used to compare against the approve/deny words.

**3. OpenCode had no idea what the conversation was about.** (This was the open item in the README's TODO list.)
* **Before:** OpenCode received only the last message.
* **Now:** `prompt_llm_code` builds the prompt from:
  * the last 6 messages, each cut to 2000 characters (`format_recent_history`),
  * followed by the approved `code_request`.

**4. One error ended the whole session.**
* **Before:** any exception killed the program, and the in-memory conversation was lost with it. Examples: invalid JSON from the classifier, an OpenRouter rate limit, a network error.
* **Now:**
  * `main()` catches the exception, prints `Bot: something went wrong (...)`, and keeps going on the same thread.
  * `Ctrl+C` / `Ctrl+D` exit cleanly instead of printing a traceback.
  * Empty input is skipped.

**5. OpenCode failures crashed the program or were hidden.** `prompt_llm_code` now turns each of these into a readable bot reply:
* the binary not being installed (`FileNotFoundError`),
* a hang (the new `CODE_TIMEOUT_SECONDS = 600` limit),
* a non-zero exit code, which used to be ignored.

**6. The agent couldn't start offline.** `draw_mermaid_png` calls the mermaid.ink web API at startup. It's now wrapped in `try`, and on failure the program prints a note and continues.

### Small cleanups
* Driver code moved into `main()` behind `if __name__ == "__main__":`, so the graph can be imported and tested without starting the chat loop.
* Removed a stray `\n\nMessages` from the end of the RAG system prompt.
* Removed the unused `from logging import RootLogger` import.
* Fixed the approval prompt text (`we areabout` typo). It now shows the full pending request and the three choices. The old extra `Continue? Y/N:` suffix, which contradicted the "type a modification" option, is gone.
* In the commented-out Claude Code alternative, the permission mode was changed from `accept-edits` to `acceptEdits`, the value the `claude` CLI actually accepts. It now gets the same context-rich prompt as OpenCode.
* A turn with an empty answer at the approval prompt now asks again instead of being treated as a modification.

### Docs
* `README.md`: fixed the Mermaid diagram, where the review step came *after* the coding agent; it now comes before. Also:
  * documented every approve/deny word, how revisions work, the error behaviour and the OpenCode prerequisite,
  * replaced the absolute `file:///.../agent_experiments/...` links with relative ones,
  * ticked off the TODO and added the remaining known limitations.
* `main_explanation.md`: rewrote the State, `classify_intent`, `accept_coding`, `prompt_llm_code` and execution-loop sections to match the code. Also:
  * explained why resuming re-runs the node from the top,
  * documented the RAG retrieval limitation.

### How it was tested
The graph was tested with a fake LLM and a fake `subprocess.run`, so no API calls were made and OpenCode never actually ran. The tests covered:
* `no`, `N`, `deny` and `Stop` all deny, and OpenCode is never called.
* Empty answer → modification → `y`:
  * the original request is kept,
  * the revision's casing is kept,
  * the revision is stored as a `human` message,
  * the prompt includes recent history.
* A coding request never leaks into a later chat turn.
* A missing binary, a timeout and a non-zero exit each produce the right reply.
* After the classifier raises an error, the same thread still works on the next message.
* The approve and deny sets have no word in common.
* The interactive loop, driven with piped stdin, handles:
  * the offline graph drawing,
  * a deny,
  * a mid-session error,
  * a clean exit on EOF.

### Known limitations (not changed here)
* `SimpleLocalSearch` returns all 3 documents for every query (`k=3`, 3 documents), and its word matching doesn't strip punctuation or ignore common words.
* `classify_intent` only sees the latest message, so short follow-ups like "do that" may be misrouted.
* Memory is in RAM only (`InMemorySaver`).
* `__pycache__/*.pyc` files are tracked in git, and `numpy` is listed as a dependency but never used.
