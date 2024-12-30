import os
import streamlit as st
from whisper_mic.whisper_mic import WhisperMic
import speech_recognition as sr
import  streamlit_toggle as tog
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxx"
string_data = "data"

st.title("Audio to Text Converter")

search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name = "duckduckgo_search",
        func = search.run,
        description="useful for when you need to answer questions about current events or the current state of the world"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history",k=2,return_messages=True)
llm = ChatOpenAI(openai_api_key="xxxxxxxxxxxxxxxxxxxxxx", temperature=1, max_tokens= 200, model_name='gpt-3.5-turbo',streaming=True)
conversational_agent = initialize_agent(
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=memory,
    handle_parsing_errors=True,
)

sys_msg = "AI talks to users"
prompt_temp = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
conversational_agent.agent.llm_chain.prompt = prompt_temp

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

if st.sidebar.button("play"):
    my_speech = my_stt()
    my_speech_str = str(my_speech)
    print(my_speech_str)
    
    my_speech_str1 = my_speech_str.lstrip("{'alternative': [{'transcript': '")
    my_speech_str2 = my_speech_str1.rstrip("', 'confidence': 0.92365956}], 'final': True}")
    my_speech_str3 = str(my_speech_str2)

    print(my_speech_str3)
    
    st.session_state.messages.append(ChatMessage(role="user", content=my_speech_str3))
    st.chat_message("user").write(my_speech_str3)

    with st.chat_message("assistant"):
        container = st.empty()
        stream_handler = StreamHandler(container)
        st_callback = StreamlitCallbackHandler(st.container())
        response = conversational_agent.run(my_speech_str3, callbacks=[st_callback])
        st.session_state.messages.append(ChatMessage(role="assistant", content=str(response)))
        st.write(response) 
    
