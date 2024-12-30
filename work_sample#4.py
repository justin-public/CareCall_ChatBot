import streamlit as st

# 세션 상태 설정
if 'show_markdown' not in st.session_state:
    st.session_state.show_markdown = True

# markdown 제거 버튼
if st.button('Markdown 지우기'):
    st.session_state.show_markdown = False
    st.experimental_rerun()

# markdown 출력 조건
if st.session_state.show_markdown:
    st.markdown("# 이것은 큰 헤더입니다.")
