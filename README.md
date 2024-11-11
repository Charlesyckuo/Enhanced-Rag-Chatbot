# An AI chatbot based on Enhanced Nvidia basic Multimodoal RAG
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

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-folder>

3. Install dependencies:
```bash
pip install -r requirements.txt

4. Run the application:
```bash
streamlit run app.py

## Usage
Access the Streamlit interface via the link displayed in your terminal after running the app.
Upload files and Input queries directly; the system will process them with query decomposition, expansion, reranking, and refinement, providing well-structured answers.

