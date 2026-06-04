i made this repo to get a good grasp on agents, so here there will be only trial scripts and nothing too serious like a project but *IT WILL BE REALLY USEFUL FOR LEARNING*

1. langgraph : making a langgraph agent 
  * [x] user messages
  * [x] memory : added memory using the inmemorysaver
  * [ ] conditional edges : choosing between three "modes"
        1. chat 
        2. knowledge retrieval *(RAG*)
        3. coding *(using claude code)*
        graph:
        ```mermaid
  flowchart LR
      U[User Input] --> C[Intent Router]
  
      C -- Chat --> P[LLM Prompting]
      C -- Knowledge --> R[RAG System]
      C -- Coding --> CC[Claude Code Agent]
  
      CC --> HIL[Human-in-the-Loop Review]
  
      P --> E[Response]
      R --> E
      HIL --> E
      ```
