import streamlit as st
import os
import openai
import tiktoken
import pytesseract
import base64
import tempfile
import uuid
import io
import re

from PIL import Image
from io import BytesIO
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, CompositeElement
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from IPython.display import HTML, display
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from PIL import Image

# Streamlit 페이지 설정
st.set_page_config(page_title="PDF 파일 챗봇", page_icon=":robot:")

# 사이드바에서 파일 업로드 및 OpenAI API 키 입력
st.sidebar.title("PDF 파일 챗봇")
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type=["pdf"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# OpenAI API 설정
if api_key:
    openai.api_key = api_key

# PDF 파일의 요소들을 추출
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # 추출된 요소들에서 텍스트와 테이블을 분류
    raw_pdf_elements = extract_pdf_elements(temp_file_path, os.path.basename(temp_file_path))
    texts, tables = categorize_elements(raw_pdf_elements)

    # 추출된 텍스트들을 특정 크기의 토큰으로 분할
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)

    # 텍스트, 테이블 요약 가져오기
    text_summaries, table_summaries = generate_text_summaries(
        texts_4k_token, tables, summarize_texts=True
    )

    # 이미지 요약 실행
    img_base64_list, image_summaries = generate_img_summaries(os.path.dirname(temp_file_path))

    vectorstore = Chroma(
        collection_name="sample-rag-multi-modal", embedding_function=OpenAIEmbeddings(api_key=api_key)
    )

    # 검색기 생성
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )

    # RAG 체인 생성
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)

    # 사용자 인터페이스
    st.title("PDF 파일 챗봇")
    st.write("PDF 파일에서 추출한 텍스트와 이미지를 바탕으로 질문에 답변합니다.")

    user_question = st.text_input("질문을 입력하세요:")
    if user_question:
        with st.spinner("답변을 생성하는 중..."):
            docs = retriever_multi_vector_img.vectorstore.similarity_search(user_question, k=5)
            answer = chain_multimodal_rag.run({"context": docs, "question": user_question})
            st.write("답변:", answer)
else:
    st.write("PDF 파일을 업로드하고 OpenAI API 키를 입력하세요.")
