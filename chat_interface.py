from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from langchain.chains.summarize import load_summarize_chain
from nltk.tokenize import sent_tokenize
import fitz
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import json
import os
import re
from unidecode import unidecode
from datasets import load_dataset
import torch
from transformers import pipeline
import soundfile as sf
import subprocess
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
import uuid
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from nltk.tokenize import sent_tokenize
import os
import json
from datetime import datetime
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import ast
import streamlit as st

with open("openai_config.json") as f:
    key = json.load(f)

os.environ["OPENAI_API_KEY"] = key["openai_key"]

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

def generate_unique_id():
    """
    Generate a unique identifier using UUID.

    Returns:
    str: A unique identifier.
    """
    return str(uuid.uuid4())


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class Chatbot():
    def __init__(self):
        with open("openai_config.json") as f:
            key = json.load(f)

        self.memory = []
        self.system_prompt = "You are a helpful assistant that helps answers questions about a pdf based on the context provided."
        self.instruction = "Please answer the following question in less than 100 words. Use ony the context provided to generate the answer: "
        self.query = ""
        self.chat_model = ChatOpenAI(temperature=0, openai_api_key=key["openai_key"])
        self.model = SentenceTransformer("all-mpnet-base-v2")
        huggingface_ef = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        self.vdb = vdb_create(collection_name="ResearchDB")
        vdb(self.vdb)
        self.embeddings_dataset = load_dataset(r"Matthijs\cmu-arctic-xvectors.py", split="validation")
        self.audio_embedding = self.get_speaker_embedding(self.embeddings_dataset, 7306)
        self.synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

    def context_retrieval(self, user_input):
        no_of_docs = 20 
        retriever = self.vdb.as_retriever(search_kwargs={"k": no_of_docs})
        compressor = CohereRerank(cohere_api_key="dM2HuLIOkp4M0cE1rOQzudjSX4bE9PWteMP2c5MV")
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

        source_df = pd.DataFrame(columns=['Text','id','title','relevance_score'])
        compressed_docs = compression_retriever.get_relevant_documents(user_input)

        for i in range(3):
            source_df = source_df.append({'Text': compressed_docs[i].page_content,
                                        'id': compressed_docs[i].metadata['id'],
                                        'Date': compressed_docs[i].metadata['Date'],
                                        'relevance_score': compressed_docs[i].metadata['relevance_score']}, ignore_index=True)    

        return ', '.join(list(source_df['Text'].unique()))    

    def get_speaker_embedding(self, embeddings_dataset, index):
        return torch.tensor(embeddings_dataset[index]["xvector"]).unsqueeze(0)

    def speech_generation(self, response):
        result = self.synthesizer(response, forward_params={"speaker_embeddings": self.audio_embedding})
        sf.write(r"chatbot_response\response.wav", result["audio"], samplerate=22050)


    def generate_response(self,
                          user_input,
                          context):
        messages = [
            SystemMessage(
                content=self.system_prompt
            ),
            HumanMessage(
                content=f"{self.instruction} User Query: {user_input} Context to answer the question: {context}"
            ),
        ]

        response = self.chat_model(messages)
        return response

    def receive_input(self, user_input):
        self.memory.append({"user": user_input})
        #qa_dataset = pd.read_csv(r'qa_dataset\QA.csv')
        #context = self.cosine_score_compute(user_input, qa_dataset, self.model)
        context = self.context_retrieval(user_input)
        response = self.generate_response(user_input, context)
        self.memory.append({"bot": response})
        #self.speech_generation(response.content)
        return response.content
    
# Define a global variable to hold the Chatbot instance
if 'chatbot_instance' not in st.session_state:
    st.session_state.chatbot_instance = Chatbot()

# Start a form for user message input
with st.form("user_input_form"):
    # Text input for the user message within the form
    user_input = st.text_input("Message:", key="user_input", placeholder="Type your message here...")

    # Button to send the message within the form
    submitted = st.form_submit_button('Send')

# Logic to handle when the message is sent
if submitted:
    if user_input.lower().strip() in ["thank you", "that's all for now", "goodbye"]:
        st.session_state.conversation.append(("You", user_input))
        st.session_state.conversation.append(("Bot", "You're welcome! Feel free to ask me anything anytime."))
        st.info("Conversation ended. Refresh the page to start a new conversation.")
    else:
        # Add user message to the conversation
        st.session_state.conversation.append(("You", user_input))

        # Generate and add bot response to the conversation
        response = st.session_state.chatbot_instance.receive_input(user_input)
        st.session_state.conversation.append(("Bot", response))

# Display the conversation in a fancier way
st.write("### Conversation")
for author, message in st.session_state.conversation:
    # Check the author to apply different styling
    if author == "You":
        # Display user messages on the left
        st.container().markdown(f"**You**: {message}", unsafe_allow_html=True)
    else:
        # Display bot messages on the right with a different color
        st.container().markdown(f"<div style='background-color:#f0f2f6;padding:10px;border-radius:10px;margin-bottom:10px;'>**Bot**: {message}</div>", unsafe_allow_html=True)