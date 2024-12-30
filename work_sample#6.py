import streamlit as st
from langchain.schema import ChatMessage
import pyautogui
import time
markdown_user_output = st.empty()
chat_input_container = st.empty()

# URL의 쿼리 파라미터 가져오기
query_params = st.experimental_get_query_params()

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="", content="")]
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if query := chat_input_container.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=query))
    with st.chat_message("user"):
        markdown_user_output.markdown(query)
    
if st.button('웹 초기화'):
    pyautogui.keyDown('ctrl')
    time.sleep(2)
    pyautogui.press('r', interval=0.5)
    pyautogui.keyUp('ctrl')

    #pyautogui.press('ctrl', interval=1)
    #pyautogui.press('r', interval=0.2)

        
              


# 웹 초기화 버튼
#if st.button('웹 초기화'):
    # 쿼리 파라미터를 변경 (또는 제거)하여 페이지 새로고침
    #st.experimental_set_query_params()
    #markdown_test.markdown("지워져??")
    #markdown_user_output.empty()
