import os
import requests
import base64
from openai import OpenAI

# Query Decomposition using OpenAI API
def query_decomposition(query) -> tuple[str, str]:
    """Decompose a complex query into two simpler sub-queries."""
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    # Prompt for query decomposition
    prompt = (
        f"You are an expert in query decomposition. Decompose the following complex query into two smaller, manageable sub-queries: \"{query}\". "
        f"If the original query is simple enough, you can expand it into two manageable queries. The two queries should be separated by a newline (\\n) without numbering. "
        f"You do not have to answer both queries! Just give me the queries only, without phrases like 'here is your decomposed query'."
    )

    # Request the response from OpenAI
    completion = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    
    # Extract response content
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content

    # Split response text into questions
    questions = response_text.strip().split("\n")

    # Assign two sub-queries if available
    if len(questions) == 2:
        question1, question2 = questions[0].strip(), questions[1].strip()
    else:
        raise ValueError("Response does not contain exactly two questions.")

    return question1, question2

# Response Combination to create a comprehensive answer
def response_combination(response1, response2) -> str:
    """Combine two responses into a comprehensive, well-structured response."""
    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    # Initialize OpenAI client
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )

    # Prompt to combine responses
    prompt = (
        "You are an expert in summarizing and synthesizing information. Your task is to combine the following two responses into a single, "
        "well-organized, comprehensive response that is succinct, clear, and includes all the important points. "
        "Avoid redundancy and ensure that the key points from both responses are well represented. Here are the two responses:\n\n"
        f"Response 1:\n{response1}\n\n"
        f"Response 2:\n{response2}\n\n"
        "Provide a complete understanding of the information presented in both responses without losing any critical details. "
        "Use well-defined paragraphs, adding line breaks to indicate different points for easier reading. "
        "Do not include any phrases like 'Here is the combined response:'. Just give me the answer only!"
    )

    # Request the combined response from OpenAI
    completion = client.chat.completions.create(
        model="meta/llama3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        top_p=1,
        max_tokens=1024,
        stream=True
    )
    
    # Extract response content
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content

    return response_text