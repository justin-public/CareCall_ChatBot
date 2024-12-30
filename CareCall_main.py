# 2023.08.11
import os
import streamlit as st
import speech_recognition as sr
#from whisper_mic.whisper_mic import WhisperMic
from io import StringIO
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from gtts import gTTS
import playsound
import pickle
#import  streamlit_toggle as tog
import pyautogui
import time

os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxx"

stt_message_check = 0
tts_message_check = 0
option_version1 = "Type#1: 일상대화 버전 (전화 받음)"

st.set_page_config(page_title="EPICGRAM DemoChatbot #1", page_icon=":robot:")
st.sidebar.header("EPICGRAM DemoChatbot #1")

text_input_container = st.empty()
place_holder_option = st.sidebar.empty()
place_holder_stt = st.sidebar.empty()
place_holder_tts = st.sidebar.empty()
place_holder_file = st.sidebar.empty()
place_holder_chunksize = st.sidebar.empty()
place_holder_memoryk = st.sidebar.empty()
place_holder_temp = st.sidebar.empty()

with open('chunk_save.txt','rb') as f:
    chunksize_number_copy = pickle.load(f)
    chunksize_number_copy_int = int(float(chunksize_number_copy))

with open('conversation_memory.txt','rb') as f:
    memory_k_number_copy = pickle.load(f)
    memory_k_number_copy_int = int(float(memory_k_number_copy))

with open('ChatTemp.txt','rb') as f:
    ChatOpenAI_number_copy = pickle.load(f)
    ChatOpenAI_number_copy_float = float(ChatOpenAI_number_copy)

with open('TokenTemp.txt','rb') as f:
    ChatOpenAI_tokens_number_copy = pickle.load(f)
    ChatOpenAI_tokens_number_copy_int = int(float(ChatOpenAI_tokens_number_copy))

with open('OnePrompt.txt','rb') as f:
    One_prompt_temp_indicator_copy = pickle.load(f)

with open('Chatting_counter','rb') as f:
    Counter_number_copy = pickle.load(f)
    Counter_number_copy_int = int(float(Counter_number_copy))

with open('TwoPrompt.txt','rb') as f:
    Two_prompt_temp_indicator_copy = pickle.load(f)

with open('ThreePrompt.txt','rb') as f:
    Three_prompt_temp_indicator_copy = pickle.load(f)

with open('FourPrompt.txt','rb') as f:
    Four_prompt_temp_indicator_copy = pickle.load(f)

with open('FivePrompt.txt','rb') as f:
    Five_prompt_temp_indicator_copy = pickle.load(f)

def sidebar_selector():
    option = place_holder_option.selectbox(
        'Type',
        ('Type#1: 일상대화 버전 (전화 받음)','Type#2: 인생기록 버전(전화 받음)', 'Type#3: 궁금증해결 버전 (전화 걸다)')
    )
    return option

def check_stt():
   stt_message = place_holder_stt.checkbox('STT')
   return stt_message

def check_tts():
    tts_message = place_holder_tts.checkbox('TTS')
    return tts_message

def file_load():
    uploaded_file = place_holder_file.file_uploader("파일 열기")
    return uploaded_file

def Textsplit_chunksize():
    chunksize_data = place_holder_chunksize.number_input('TextSplitter의 Chunk_size 값',value=chunksize_number_copy_int)
    return chunksize_data

def Conversation_memory_K():
    memory_k = place_holder_memoryk.number_input('ConversationBufferWindowMemory의 K값',value=memory_k_number_copy_int)
    return memory_k

def ChatOpenAI_temp_value():
    ChatOpenAI_data = place_holder_temp.number_input('ChatOpenAI의 temperature값',value=ChatOpenAI_number_copy_float)
    return ChatOpenAI_data

def ChatOpenAI_max_tokens_value():
    ChatOpenAI_tokens_data = st.sidebar.number_input('ChatOpenAI의 max_tokens값',value=ChatOpenAI_tokens_number_copy_int)
    return ChatOpenAI_tokens_data

def OneSet_prompt():
    One_input_prompt = st.sidebar.text_area(label="1st Prompt template",value=One_prompt_temp_indicator_copy,height=100,placeholder=None)
    return One_input_prompt

def Prompt_count():
    count_data = st.sidebar.number_input('1st to 2nd Prompt로 변경될 질문 횟수값',value=Counter_number_copy_int) #Counter_number_copy_int
    return count_data

def TwoSet_prompt():
    Two_input_prompt = st.sidebar.text_area(label="2nd Prompt template",value=Two_prompt_temp_indicator_copy,height=100,placeholder=None)
    return Two_input_prompt

def ThreeSet_prompt():
    Three_input_prompt = st.sidebar.text_area(label="3nd Prompt template",value=Three_prompt_temp_indicator_copy,height=100,placeholder=None)
    return Three_input_prompt

def FourSet_prompt():
    Four_input_prompt = st.sidebar.text_area(label="4nd Prompt template",value=Four_prompt_temp_indicator_copy,height=100,placeholder=None)
    return Four_input_prompt

def FiveSet_prompt():
    Five_input_prompt = st.sidebar.text_area(label="5nd Prompt template",value=Five_prompt_temp_indicator_copy,height=100,placeholder=None)
    return Five_input_prompt

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="", content="")]
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

def my_stt() :
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("무슨 말이라도? : ")
        audio = r.listen(source)
    mySpeech = r.recognize_google(audio, language='ko')  # show_all=True
    try :
        return mySpeech
    except sr.UnknownValueError:
        print("Google 음성 인식이 오디오를 이해할 수 없습니다.")
    except sr.RequestError as e:
        print("Google 음성 인식 서비스에서 결과를 요청할 수 없습니다.; {0}".format(e))

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename='voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

#def init_load():
    #with open('chunk_save.txt','rb') as f:
        #chunksize_number_copy = pickle.load(f)
        #chunksize_number_copy_int = int(float(chunksize_number_copy))
    #return chunksize_number_copy_int   


def main():
    string_data = "data"
    option_type = sidebar_selector()
    stt_message_check = check_stt()    
    my_speech_str = ""
    tts_message_check = check_tts()
    
    uploaded_file_data = file_load()
    if uploaded_file_data is not None:
        bytes_data = uploaded_file_data.getvalue()
        stringio = StringIO(uploaded_file_data.getvalue().decode("utf-8"))
        string_data = stringio.read()
    
    chunksize_value = Textsplit_chunksize()
    memory_k_value = Conversation_memory_K()
    temp_value = ChatOpenAI_temp_value()
    token_value=ChatOpenAI_max_tokens_value()
    One_input_prompt_indicator = OneSet_prompt()
    Prompt_count_value = Prompt_count()
    Two_input_prompt_indicator=TwoSet_prompt()
    Three_input_prompt_indicator=ThreeSet_prompt()
    Four_input_prompt_indicator=FourSet_prompt()
    Five_input_prompt_indicator=FiveSet_prompt()

    col1, col2 = st.sidebar.columns([1,1])
    with col1:
        if st.button("통화 시작", kwargs={'clicked_button_ix': 1, 'n_buttons': 4}):
            with open('chunk_save.txt','wb') as f:
                pickle.dump(str(chunksize_value),f)
            with open('conversation_memory.txt','wb') as f:
                pickle.dump(str(memory_k_value),f)
            with open('ChatTemp.txt','wb') as f:
                pickle.dump(temp_value,f)
            with open('TokenTemp.txt','wb') as f:
                pickle.dump(token_value,f)
            with open('OnePrompt.txt','wb') as f:
                pickle.dump(One_input_prompt_indicator,f)
            with open('Chatting_counter','wb') as f:
                pickle.dump(Prompt_count_value,f)
            with open('TwoPrompt.txt','wb') as f:
                pickle.dump(Two_input_prompt_indicator,f)
            with open('ThreePrompt.txt','wb') as f:
                pickle.dump(Three_input_prompt_indicator,f)
            with open('FourPrompt.txt','wb') as f:
                pickle.dump(Four_input_prompt_indicator,f)
            with open('FivePrompt.txt','wb') as f:
                pickle.dump(Five_input_prompt_indicator,f)    

    
    with col2:
        if st.button("통화 종료", kwargs={'clicked_button_ix': 2, 'n_buttons': 4}):
            with open("conversation.txt", "w") as file:
                for line in st.session_state.conversation_list:
                    file.write(line + "\n")
            st.success("대화 내용이 conversation.txt 파일에 저장되었습니다.")

    text_splitter = CharacterTextSplitter(chunk_size=chunksize_number_copy_int, chunk_overlap=0)
    texts = text_splitter.split_text(string_data)
    embeddings = OpenAIEmbeddings(openai_api_key="xxxxxxxxxxxxx")
    docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))])
    
    if 'count' not in st.session_state:
        st.session_state['count'] = 0
    
    if 'count1' not in st.session_state:
        st.session_state['count1'] = 0
    
    if 'my_string' not in st.session_state:
        st.session_state.my_string = ""
    
    if "conversation_list" not in st.session_state:
        st.session_state.conversation_list = []  

    if 'show_markdown' not in st.session_state:
        st.session_state.show_markdown = True

    if st.sidebar.button("채팅창 Reset"):
        pyautogui.keyDown('ctrl')
        time.sleep(2)
        pyautogui.press('r', interval=0.5)
        pyautogui.keyUp('ctrl')
    
    if stt_message_check == True:
        text_input_container.empty()
        if st.button(":microphone:",help='마이크를 이용해 질문해 보세요'):
            my_speech = my_stt()
            my_speech_str = str(my_speech)
        
        if len(my_speech_str) > 0:
            st.session_state.messages.append(ChatMessage(role="user", content=my_speech_str))
            st.chat_message("user").write(my_speech_str)
            st.session_state.conversation_list.append(str("User : "+ my_speech_str))
            
            with st.chat_message("assistant"):
                st.session_state['count'] += 1
                count_chk = int(st.session_state['count'])
                count_chk_temp = count_chk - 1
                
                st.write(count_chk_temp)
                
                if count_chk_temp == 0:
                    st.session_state.my_string = One_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                
                if count_chk_temp == 1 and count_chk_temp < Counter_number_copy_int + 1:
                    st.session_state.my_string = Two_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                
                if count_chk_temp == Counter_number_copy_int + 1:
                    st.session_state.my_string = Three_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                     
                if count_chk_temp == Counter_number_copy_int + 2 and count_chk_temp < Counter_number_copy_int + Counter_number_copy_int + 2:
                    st.session_state.my_string = Four_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                    
                if count_chk_temp == Counter_number_copy_int + Counter_number_copy_int + 2:
                    st.session_state.my_string = Five_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                
                template = st.session_state.my_string + """
                {context}
                {chat_history}
                Human: {human_input}
                Chatbot:"""
                prompt_temp = PromptTemplate(
                    input_variables=["chat_history", "human_input", "context"],
                    template=template
                )
                
                memory = ConversationBufferMemory(memory_key="chat_history", k=int(memory_k_number_copy_int) ,input_key="human_input")
                container = st.empty()
                stream_handler = StreamHandler(container)
                chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo',temperature=ChatOpenAI_number_copy_float,max_tokens=ChatOpenAI_tokens_number_copy_int,streaming=True,callbacks=[stream_handler]),chain_type="stuff",memory=memory, prompt=prompt_temp)
                docs = docsearch.similarity_search(my_speech_str)
                value = chain({"input_documents": docs, "human_input": my_speech_str}, return_only_outputs=True)
                value1 = str(value).lstrip('{')
                value2 = value1.rstrip('}')
                value_output = value2.split(':')
                value_output1 = str(value_output[1]).lstrip('"')
                value_output2 = value_output1.replace('"','')
                
                if tts_message_check == False:
                    st.session_state.conversation_list.append(str("AI : "+value_output2))
                    st.session_state.messages.append(ChatMessage(role="assistant", content=str(value_output2)))
                    container.markdown(str(value_output2))

                    #if count_chk_temp > Counter_number_copy_int + Counter_number_copy_int:
                        #text_input_container.empty()

                if tts_message_check == True:
                    speak(str(value_output2))
                    st.session_state.conversation_list.append(str("AI : "+value_output2))
                    st.session_state.messages.append(ChatMessage(role="assistant", content=str(value_output2)))
                    container.markdown(str(value_output2))

                    #if count_chk_temp > Counter_number_copy_int + Counter_number_copy_int:
                        #text_input_container.empty()
                    
    if stt_message_check == False:
        if query := text_input_container.chat_input():
            st.session_state.messages.append(ChatMessage(role="user", content=query))
            st.chat_message("user").markdown(query)
            st.session_state.conversation_list.append(str("User : "+query))
            
            with st.chat_message("assistant"):
                st.session_state['count'] += 1
                count_chk = int(st.session_state['count'])
                count_chk_temp = count_chk - 1
                
                st.write(count_chk_temp)  
                if count_chk_temp == 0:
                    st.session_state.my_string = One_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                
                if count_chk_temp == 1 and count_chk_temp < Counter_number_copy_int + 1:
                    st.session_state.my_string = Two_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                
                if count_chk_temp == Counter_number_copy_int + 1:
                    st.session_state.my_string = Three_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                     
                if count_chk_temp == Counter_number_copy_int + 2 and count_chk_temp < Counter_number_copy_int + Counter_number_copy_int + 2:
                    st.session_state.my_string = Four_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                    
                if count_chk_temp == Counter_number_copy_int + Counter_number_copy_int + 2:
                    st.session_state.my_string = Five_prompt_temp_indicator_copy
                    print(st.session_state.my_string)
                
                template = st.session_state.my_string + """
                {context}
                {chat_history}
                Human: {human_input}
                Chatbot:"""
                prompt_temp = PromptTemplate(
                    input_variables=["chat_history", "human_input", "context"],
                    template=template
                )
                
                memory = ConversationBufferMemory(memory_key="chat_history", k=int(memory_k_value) ,input_key="human_input")
                container = st.empty()
                stream_handler = StreamHandler(container)
                chain = load_qa_chain(OpenAI(model_name='gpt-3.5-turbo',temperature=1,max_tokens=200,streaming=True,callbacks=[stream_handler]),chain_type="stuff",memory=memory, prompt=prompt_temp)
                docs = docsearch.similarity_search(query)
                value = chain({"input_documents": docs, "human_input": query}, return_only_outputs=True)
                value1 = str(value).lstrip('{')
                value2 = value1.rstrip('}')
                value_output = value2.split(':')
                value_output1 = str(value_output[1]).lstrip('"')
                value_output2 = value_output1.replace('"','')
                
                if tts_message_check == False:
                    st.session_state.conversation_list.append(str("AI : "+value_output2))
                    st.session_state.messages.append(ChatMessage(role="assistant", content=str(value_output2)))
                    container.markdown(str(value_output2))
                    
                    #if count_chk_temp > Counter_number_copy_int + Counter_number_copy_int:
                        #text_input_container.empty()
                
                if tts_message_check == True:
                    speak(str(value_output2))
                    st.session_state.conversation_list.append(str("AI : "+value_output2))
                    st.session_state.messages.append(ChatMessage(role="assistant", content=str(value_output2)))
                    container.markdown(str(value_output2))
                    
                    # Option
                    #if count_chk_temp > Counter_number_copy_int + Counter_number_copy_int:
                        #text_input_container.empty()
                        
    if option_type == option_version1:
        print(option_type)

if __name__=="__main__":
    main()