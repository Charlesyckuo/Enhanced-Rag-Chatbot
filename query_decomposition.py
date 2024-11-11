import os
import requests
import base64
from openai import OpenAI

def query_decomposition(query) -> tuple[str, str]:

    api_key = "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"
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = api_key
    )

    prompt = f"You are an expert in query decomposition. Decompose the following complex query into two smaller, manageable sub-queries: \"{query}\". If the original query is simply enough, you can expand the query into two manageable queries. The two queries should be separated by a newline (\n) without numbering. You do not have to answer both of the query! Just give me the queries only, you do not have to say terms like 'here is your decomposed query'. Example Input: What is RAG and How can RAG enhance LLM。Example Output: What is RAG? How can Rag enhance LLM?"

    completion = client.chat.completions.create(
      model="meta/llama3-70b-instruct", 
      messages=[{"role":"user","content":prompt}],
      temperature=0.5,
      top_p=1,
      max_tokens=1024,
      stream=True
    )
    
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
    print(response_text)


    # 使用 split 方法以換行符號分割
    questions = response_text.split("\n")

    # 假設有兩個問題，分別將它們賦值為 question1 和 question2
    if len(questions) == 2:
        question1, question2 = questions[0].strip(), questions[1].strip()
    else:
        raise ValueError("Response does not contain exactly two questions.")

    # 回傳為 (question1, question2) 的形式
    print((question1, question2))
    
    return (question1, question2)

def response_combination(response1, response2) ->str:
    api_key = "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"
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = api_key
    )

    prompt = f"""
    You are an expert in summarizing and synthesizing information. Your task is to combine the following two responses into a single, well-organized, comprehensive response that is succinct, clear, and includes all the important points. Please avoid redundancy and ensure that the key points from both responses are well represented. Here are the two responses:

    Response 1:
    {response1}

    Response 2:
    {response2}

    Your output should provide a complete understanding of the information presented in both responses, without losing any critical details, and should be easy for the reader to follow. Make sure to connect the content in a logical flow. Additionally, please structure the combined response into well-defined paragraphs, using line breaks to indicate different points for easier reading. Do not include any phrases like "Here is the combined response:" in your answer. Just give me the answer only!
    """
    completion = client.chat.completions.create(
      model="meta/llama3-70b-instruct", 
      messages=[{"role":"user","content":prompt}],
      temperature=0.5,
      top_p=1,
      max_tokens=1024,
      stream=True
    )
    
    response_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response_text += chunk.choices[0].delta.content
    print(response_text)
    return response_text