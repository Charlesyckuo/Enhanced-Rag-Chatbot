import streamlit as st
from llama_index.core import Settings, VectorStoreIndex, StorageContext, get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.postprocessor import LLMRerank
import nest_asyncio

# Apply nest_asyncio to avoid async loop issues
nest_asyncio.apply()

from document_processor import load_multimodal_data
from function import set_environment_variables
from query_decomposition import query_decomposition, response_combination

# Set up the page configuration
st.set_page_config(layout="wide")

# Initialize settings
def initialize_settings():
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/llama-3.2-nv-embedqa-1b-v1", truncate="END")  # Embedding setting
    Settings.llm = NVIDIA(model="meta/llama-3.1-70b-instruct")
    Settings.text_splitter = SentenceSplitter(chunk_size=600)

# Create Milvus Vector Database
def create_vector_store():
    vector_store = MilvusVectorStore(
        host="127.0.0.1",
        port=19530,
        dim=2048,
        collection_name="temp_collection"
    )
    return vector_store

# Create Vector Database Index
def create_index(documents, vector_store):
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Main function
def main():
    set_environment_variables()
    initialize_settings()

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.title("Nvidia-Enhanced Multimodal RAG Chatbot")

        # Create Vector Database if it doesn't exist
        if 'vector_store' not in st.session_state:
            st.session_state['vector_store'] = create_vector_store()
        vector_store = st.session_state['vector_store']
        
        # Create Knowledge Base if it doesn't exist
        if 'all_documents' not in st.session_state:
            st.session_state['all_documents'] = []

        uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True)
        if uploaded_files and st.button("Process Files"):
            with st.spinner("Processing files..."):
                new_documents = load_multimodal_data(uploaded_files)
                st.session_state['all_documents'].extend(new_documents)
                
                # Rebuild index with updated documents
                st.session_state['index'] = create_index(st.session_state['all_documents'], vector_store)
                st.success("Files processed and index updated!")

        # Button to clear all uploaded data
        if st.button("Clear All Uploaded Data"):
            if 'vector_store' in st.session_state:
                st.session_state['vector_store'] = []
                st.session_state.pop('index', None)
                st.session_state.pop('all_documents', None)
                st.success("All uploaded data cleared from vector store.")
    
    with col2:
        if 'index' in st.session_state:
            st.title("Chat")

            if 'history' not in st.session_state:
                st.session_state['history'] = []
                
            query_engine = st.session_state['index'].as_query_engine(
                similarity_top_k=10,  
                streaming=True,
                node_postprocessors=[
                    LLMRerank(
                        choice_batch_size=5,
                        top_n=3,
                    )
                ],
                response_mode="refine",
            )
            user_input = st.chat_input("Ask me anything:")

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append({"role": "user", "content": user_input})

                question1, question2 = query_decomposition(user_input)

                response1 = query_engine.query(question1)
                response2 = query_engine.query(question2)
                
                full_response = response_combination(response1, response2)

                # Print out final response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    current_text = ""
                    for char in full_response:
                        current_text += char
                        message_placeholder.markdown(current_text + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state['history'].append({"role": "assistant", "content": full_response})

            # Clear Chat history button
            if st.button("Clear Chat"):
                st.session_state['history'] = []
                st.rerun()

if __name__ == "__main__":
    main()
