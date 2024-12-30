import os
import streamlit as st
import speech_recognition as sr
from whisper_mic.whisper_mic import WhisperMic
from io import StringIO
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.docstore.document import Document
#from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from gtts import gTTS
import playsound
import pickle
import  streamlit_toggle as tog
import pyautogui
import time
import PyPDF2

os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxx"

st.set_page_config(page_title="EPICGRAM DemoChatbot #1", page_icon=":robot:")
st.sidebar.header("EPICGRAM DemoChatbot #1")

place_holder_file = st.sidebar.empty()


def file_load():
    uploaded_file = place_holder_file.file_uploader("파일 열기",accept_multiple_files=True,type=["txt","pdf"])
    return uploaded_file

def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        #print("Page Number:", len(pdfReader.pages))
        for i in range(len(pdfReader.pages)):
          pageObj = pdfReader.pages[i]
          text = pageObj.extract_text()
          pageObj.clear()
          text_list.append(text)
          sources_list.append(file.name + "_page_"+str(i))
    return [text_list,sources_list]

def main():
    uploaded_file_data = file_load()
    if uploaded_file_data is not None:
        textify_output = read_and_textify(uploaded_file_data)
        #bytes_data = uploaded_file_data.getvalue()
        #stringio = StringIO(uploaded_file_data.getvalue().decode("utf-8"))
        #string_data = stringio.read()
        documents = textify_output[0]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

if __name__=="__main__":
    main()