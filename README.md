# Enhanced AI Chatbot with NVIDIA Multimodal RAG

This project builds upon NVIDIA's multimodal Retrieval-Augmented Generation (RAG) framework, integrating advanced query processing and user experience improvements. The system leverages multiple state-of-the-art models for embedding, reranking, and handling charts/images, implemented using NVIDIA's NIM microservices.

---

## Features

### **Models and Functions**
- **Chart and Image Processing**:
  - Models: `google/deplot`, `nvidia/neva-22b`
  - Implementation: Defined in `function.py`
- **Query Decomposition**:
  - Model: `meta/llama3-70b-instruct`
  - Implementation: Defined in `query_decomposition.py`
- **Embedding and Query Handling**:
  - Models: `nvidia/llama-3.2-nv-embedqa-1b-v1`, `meta/llama-3.1-70b-instruct`
  - Implementation: Defined in `app.py`

### **Key Enhancements**
- **Query Decomposition**:
  - Breaks down complex queries into sub-queries for detailed and accurate responses.
- **Automatic Query Expansion**:
  - Expands simple queries to include additional relevant context.
- **LLM Reranker**:
  - Ranks and filters top K results to prioritize the most relevant information.
- **Iterative Refinement**:
  - Uses `response_mode = refine` to iteratively improve responses based on provided context.
- **Custom Response Template**:
  - Structures answers in a concise and professional format.

### **Knowledge Base Persistence**
- Retains uploaded data across sessions, enabling continuity in responses.
- Provides the option to clear the knowledge base for generating fresh responses without prior data.

---

## Installation

### **Prerequisites**
- Python 3.8 or higher
- NVIDIA API access (API key required for `function.py`)

### **Steps**
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up NVIDIA API Key**:
   - Open `function.py` and add your NVIDIA API key.

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. **Launch the Application**:
   - After running the app, Streamlit will provide a local URL.
   - Open the URL in your web browser.

2. **Upload Files**:
   - Upload relevant data files to build the knowledge base.

3. **Input Queries**:
   - Enter queries directly. The system will process them using the following pipeline:
     - **Query Decomposition**: Breaks down complex queries.
     - **Query Expansion**: Broadens simple queries for comprehensive results.
     - **Reranking and Refinement**: Ensures the most relevant responses are provided.

4. **View Results**:
   - Receive well-structured answers combining all relevant insights.

---

## Key Benefits
- **Enhanced Accuracy**: Query decomposition and reranking ensure precise answers.
- **User-Friendly**: Streamlined interface for effortless interaction.
- **Persistent Knowledge Base**: Seamless continuation across sessions.
- **Versatile Functionality**: Handles charts, images, and textual data effectively.

For further inquiries or support, please refer to the repository's issues section.

