from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
import os
import json
from datetime import datetime
import uuid
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from nltk.tokenize import sent_tokenize

from sentence_transformers import SentenceTransformer, util
import pandas as pd

from langchain.chains.summarize import load_summarize_chain

from nltk.tokenize import sent_tokenize

import json
import os
import re
from unidecode import unidecode
from datasets import load_dataset
import torch
from transformers import pipeline
import soundfile as sf
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import ast

def generate_unique_id():
    """
    Generate a unique identifier using UUID.

    Returns:
    str: A unique identifier.
    """
    return str(uuid.uuid4())

def pretty_print_docs(docs):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )

def generate_response(user_input):
    # Define a new response schema for extracting search terms
    chat_model = ChatOpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))  
    keyword_response_schemas = [
        ResponseSchema(
            name="List of keyword search terms for vector database",
            description="Extract relevant keywords from the user's query for vector database searches.",
            format_instructions="List all relevant keywords, separated by commas."
        )
    ]

    # Initialize the output parser with the new response schema
    keyword_output_parser = StructuredOutputParser.from_response_schemas(keyword_response_schemas)
    keyword_format_instructions = keyword_output_parser.get_format_instructions()
    keyword_system_prompt = "You are a helpful assistant that extracts key search terms from user queries for vector database searches."
    keyword_instruction = "Extract and list all relevant search terms from the user's query that should be used for a vector database search."

    query_response_schemas = [
        ResponseSchema(
            name="List of rewritten queries for vector database search",
            description="Generate new queries given a list of keywords and user query",
            format_instructions="List of rewritten queries, separated by commas."
        )
    ]

    # Initialize the output parser with the new response schema
    query_output_parser = StructuredOutputParser.from_response_schemas(query_response_schemas)
    query_format_instructions = query_output_parser.get_format_instructions()
    query_system_prompt = "You are a helpful assistant that generates a list of rewritten vector database queries given a user query and keywords for vector database searches."
    query_instruction = "Generate 5 different rewritten vector database search queries from the user's query that should be used for a vector database search."


    keywords_messages = [
        SystemMessage(
            content=keyword_system_prompt
        ),
        HumanMessage(
            content=f"{keyword_instruction} User Query: {user_input} Formatting Instructions: {keyword_format_instructions}"
        ),
    ]

    keywords = ast.literal_eval(chat_model(keywords_messages).content.replace("```json","").replace("```",""))['List of keyword search terms for vector database']
    rewritten_queries_messages = [
        SystemMessage(
            content=query_system_prompt
        ),
        HumanMessage(
            content=f"{query_instruction} User Query: {user_input} Keywords: {keywords} Formatting Instructions: {query_format_instructions}"
        ),
    ]
    queries = ast.literal_eval(chat_model(rewritten_queries_messages).content.replace("```json","").replace("```",""))['List of rewritten queries for vector database search']
    return queries

def vdb_create(collection_name="ResearchDB"):
  client = chromadb.Client()

  huggingface_ef = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

  collections = Chroma(persist_directory="vdb", embedding_function=huggingface_ef)

  return collections

def vdb_insert(collection, folders_and_json_files):
    """
    A dummy function to apply to the contents of each JSON file.
    You can replace this with your actual processing logic.
    """
    # For demonstration purposes, this function simply prints the data.
    huggingface_ef = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    for folder_name, json_file_list in folders_and_json_files.items():
        for json_file in json_file_list:
            print("Insertion Begun")
            with open(os.path.join(f"linkedin_posts/{folder_name}", json_file), 'r') as f:
                data = json.load(f)
                print(list(data.keys()))
                # algo = data['Algorithmic Innovations']
                # theo_ana = data['Theoretical Analyses']
                # empi_res = data['Empirical Results']
                # apps_n_use = data['Applications and Use Cases']
                # ds_n_preproc = data['Datasets and Data Preprocessing']
                # ethic_n_soc = data['Ethical and Societal Implications']
                # fut_dir_n_open_prob = data['Future Directions and Open Problems']
                # eng = data['Engagement']
                docs = []
                for chunk in splitter.split_text(data['Summary']):
                    print(chunk)
                    docs.append(Document(page_content=chunk, metadata={"id": str(generate_unique_id()),
                                                                       "Date": folder_name,
                                                                       "Summary": data['Summary']}))
            vector_collection = Chroma.from_documents(documents=docs, persist_directory="vdb", embedding=huggingface_ef)
            #collection.extend(vector_collection)  # Adding vector_collection to the main collection
    print("All Records have been inserted into Vector Database")   


def vdb(collection):
    """
    Get a list of all folders in the specified parent folder and process
    JSON files in each folder.

    Args:
    parent_folder (str): Path to the parent folder.

    Returns:
    dict: A dictionary mapping folder names to lists of JSON file names.
    """
    parent_folder = f"./linkedin_posts"
    folders_and_json_files = {}
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        if os.path.isdir(folder_path):
            folders_and_json_files[folder] = []
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
            folders_and_json_files[folder].extend(json_files)
    vdb_insert(collection, 
               folders_and_json_files)
    print("All Records have been inserted into Vector Database")
    #return folders_and_json_files


huggingface_ef = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
collection = vdb_create(collection_name="ResearchDB")
vdb(collection)
vectordb = Chroma(persist_directory="vdb", embedding_function=huggingface_ef)
#folders_and_json_files = get_folders_and_json_files("2024-03-19", collection)
#print(get_folders_and_json_files(collection))

#results = collection.query(query_texts="'Find latest large language model designs'",
#where = sample_where_clause,
#n_results=2)

#context = results['documents']
#print(context)

no_of_docs = 20 
retriever = collection.as_retriever(search_kwargs={"k": no_of_docs})
compressor = CohereRerank(cohere_api_key="dM2HuLIOkp4M0cE1rOQzudjSX4bE9PWteMP2c5MV")
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
)

source_df = pd.DataFrame(columns=['Text','id','title','relevance_score'])

user_input = "What is the model architecture of Qwen?"
compressed_docs = compression_retriever.get_relevant_documents(user_input)

docs = retriever.get_relevant_documents("Text to video")

print("Base query, Compressed")
pretty_print_docs(compressed_docs)

for i in range(3):
    source_df = source_df.append({'Text': compressed_docs[i].page_content,
                                  'id': compressed_docs[i].metadata['id'],
                                  'Date': compressed_docs[i].metadata['Date'],
                                  'relevance_score': compressed_docs[i].metadata['relevance_score']}, ignore_index=True)

    
source_df.to_csv(r"context_retrieved/compressed_context.csv")

response = generate_response(user_input)

for response in response[0].split(","):
    print("Reformatted Query: "+response)
    docs = retriever.get_relevant_documents(response)
    print("Base Retriver: ")
    pretty_print_docs(docs)
    compressed_docs = compression_retriever.get_relevant_documents(user_input)
    print("Compressed Retriver: ")
    pretty_print_docs(compressed_docs)