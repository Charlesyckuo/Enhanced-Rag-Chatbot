# Multimodal RAG with NVIDIA NIM Microservices
This project extends NVIDIA's multimodal Retrieval-Augmented Generation (RAG) framework with advanced query processing and user experience enhancements. It incorporates multiple models for embedding, reranking, and chart/image handling, using NVIDIA's NIM microservices.

## Features
### Models and Functions Used
- "google/deplot" and "nvidia/neva-22b": For chart and image processing (in function.py).
- "meta/llama3-70b-instruct": For query decomposition (in query_decomposition.py).
- "nvidia/llama-3.2-nv-embedqa-1b-v1" and "meta/llama-3.1-70b-instruct": For embedding and foundational query handling (in app.py).

### Key Enhancements
- Query Decomposition: Breaks down complex queries into sub-queries for refined answers.
- Automatic Query Expansion: Expands simple queries to capture additional relevant information.
- LLM Reranker: Ranks and filters top K results to enhance response accuracy.
- Iterative Refinement: Uses response_mode = refine to consider context iteratively, improving response quality.
- Custom Response Template: Combines responses into a well-structured answer.

### Knowledge Base Persistence
- Retains uploaded data across sessions.
- Option to clear accumulated knowledge, allowing for fresh responses without prior data.
