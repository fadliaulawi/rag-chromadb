from pinecone import Pinecone
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import umap
from helper_utils import word_wrap, project_embeddings

def single_query_expansion(query):
    index, openai_client = prerequisities()
    augmented_query = augment_query_generated(openai_client, query)
    joint_query = query + " " + augmented_query
    results = query_vector(openai_client, joint_query, index)
    return results

def multiple_query_expansion(query):
    index, openai_client = prerequisities()
    augmented_queries = augment_multiple_query(openai_client, query)
    queries = [query] + augmented_queries
    results = set()
    final = []
    
    for query in queries:
        response = index.query(
            vector=embedding_function(openai_client, query),
            top_k=2,
            include_metadata=True,
        )['matches']
        
        lst_tmp = set()
        for dicts in response:
            lst_tmp.add(dicts['metadata']['text'])
        
        text_holder = ""
        for text in lst_tmp:
            text_holder += text + "\n"
        
        results.add(text_holder)
        
    for doc in results:
        final.append(doc)
        
    return final
    
def prerequisities():
    load_dotenv()
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index = pc.Index("rag-kak")
    openai.api_key = os.environ['OPENAI_API_KEY']
    openai_client = OpenAI()
    
    return index, openai_client
    
def embedding_function(openai_client, text, model="text-embedding-ada-002"):
    
    response = openai_client.embeddings.create(
        input=text,
        model=model
    )
    
    embedding = response.data[0].embedding
    
    return embedding

def augment_query_generated(openai_client, query, model="gpt-4o"):
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable healthcare research assistant. Your users are asking questions about information contained in a healthcare document. You will be shown the user's question and the relevant information from the healthcare document. Answer the question with support of the provided document."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

def query_vector(openai_client, joint_query, index):
    results = index.query(
        vector=embedding_function(openai_client, joint_query),
        top_k=10,
        include_metadata=True
    )["matches"]
    
    tmp = []
    
    for result in results:
        tmp.append(result["metadata"]['text'])
        
    return tmp

def augment_multiple_query(openai_client, query, model="gpt-4o"):
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable healthcare research assistant. Your users are asking questions about information contained in a healthcare document."
            "Suggest up to four additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content