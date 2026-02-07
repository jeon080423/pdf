import streamlit as st
import os
import PyPDF2
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF 분석 및 Q&A 시스템",
    page_icon="📄",
    layout="wide"
)

# Initialize API Client
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    # Also check Streamlit secrets for deployment
    if not api_key and "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
    
    if api_key:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )
    return None

client = get_client()
MODEL_NAME = "openai/gpt-oss-120b"

def extract_text_from_pdf(file):
    text = ""
    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None
    return text

def analyze_report(text):
    if not client:
        st.warning("API Key가 설정되지 않았습니다. .env 파일이나 Streamlit Secrets를 확인해주세요.")
        return None

    try:
        # Truncate text for context window
        truncated_text = text[:15000]
        
        prompt = f"""
        당신은 전문 보고서 분석가입니다. 아래 제공되는 [보고서 전체 텍스트]를 읽고 다음 작업을 수행하세요:
        1. 이 보고서의 핵심 주제와 목적을 한 문장으로 요약하세요.
        2. 전체 내용을 바탕으로 5개의 주요 키워드를 추출하세요.
        3. 보고서의 전체적인 논조와 결론을 요약하여 사용자에게 인사이트를 제공할 준비를 하세요.
        
        [보고서 전체 텍스트]
        {truncated_text}
        
        응답 형식(JSON):
        {{
            "summary": "요약 내용",
            "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"]
        }}
        """
        
        completion = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return None

def get_answer(question, context):
    if not client:
        return "API Key가 설정되지 않았습니다."
        
    try:
        # Truncate context for Q&A
        cmd_context = context[:20000]
        
        prompt = f"""
        사용자가 다음 질문을 선택했습니다: {question}.
        전체 보고서의 내용을 바탕으로 가장 정확하고 신뢰할 수 있는 답변을 생성하세요.
        답변 시 반드시 다음 지침을 따르세요:
        1. 보고서 내에 근거가 있는 내용만 답변에 포함하세요.
        2. 만약 보고서에 관련 내용이 없다면 '보고서 내에서는 확인되지 않는 내용입니다'라고 명시하세요.
        3. 답변의 신뢰도를 높이기 위해 관련 내용이 위치한 보고서의 섹션이나 페이지를 언급하세요(가능한 경우).
        
        [보고서 전체 텍스트]
        {cmd_context}
        """
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error responding to question: {e}"

# UI Layout
st.title("📄 PDF 분석 및 Q&A 시스템")

with st.sidebar:
    st.header("설정 및 업로드")
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=['pdf'])
    
    if not client:
        st.error("⚠️ Groq API Key가 없습니다.")
        st.info("로컬 실행 시 .env 파일에 GROQ_API_KEY를 설정하거나, Streamlit Cloud 배포 시 Secrets에 추가하세요.")

if uploaded_file is not None:
    # Process PDF
    if 'pdf_text' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("PDF 텍스트 추출 중..."):
            text = extract_text_from_pdf(uploaded_file)
            if text:
                st.session_state.pdf_text = text
                st.session_state.current_file = uploaded_file.name
                # Reset analysis on new file
                if 'analysis_result' in st.session_state:
                    del st.session_state.analysis_result
                if 'messages' in st.session_state:
                    del st.session_state.messages
            else:
                st.error("PDF에서 텍스트를 추출할 수 없습니다.")

    if 'pdf_text' in st.session_state:
        # Create tabs
        tab1, tab2 = st.tabs(["📊 분석 결과", "💬 Q&A 채팅"])
        
        with tab1:
            st.header("보고서 분석")
            if st.button("보고서 분석 시작"):
                with st.spinner("AI가 보고서를 분석 중입니다..."):
                    result = analyze_report(st.session_state.pdf_text)
                    if result:
                        st.session_state.analysis_result = result
            
            if 'analysis_result' in st.session_state:
                res = st.session_state.analysis_result
                st.subheader("📝 요약")
                st.info(res.get('summary', '요약 없음'))
                
                st.subheader("🔑 주요 키워드")
                # Display keywords as tags
                cols = st.columns(len(res.get('keywords', [])))
                for i, keyword in enumerate(res.get('keywords', [])):
                    # Use container or just write
                    st.success(f"#{keyword}")

        with tab2:
            st.header("질문하기")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("보고서 내용에 대해 질문하세요"):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})

                with st.chat_message("assistant"):
                    with st.spinner("답변 생성 중..."):
                        response = get_answer(prompt, st.session_state.pdf_text)
                        st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드해주세요.")

