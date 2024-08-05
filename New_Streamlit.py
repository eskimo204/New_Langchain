import streamlit as st
import os
import openai
import pytesseract
import base64
import tempfile

# from PyPDF2 import PdfFileReader
from PIL import Image
from io import BytesIO
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Streamlit 페이지 설정
st.set_page_config(page_title="PDF 파일 챗봇", page_icon=":robot:")

# 사이드바에서 파일 업로드 및 OpenAI API 키 입력
st.sidebar.title("PDF 파일 챗봇")
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type=["pdf"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# OpenAI API 설정
if api_key:
    openai.api_key = api_key

def extract_text_and_images_from_pdf(file_path):
    # PDF에서 텍스트와 이미지를 추출
    elements = partition_pdf(file_path)
    text = ""
    images = []
    
    for element in elements:
        # 요소 타입을 확인하여 적절히 처리
        if isinstance(element, dict) and 'type' in element:
            if element['type'] == 'Text':
                text += element['text']
            elif element['type'] == 'Image':
                images.append(element['base64'])
    
    return text, images


def text_to_chunks(text):
    # 텍스트를 청크로 분할
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def encode_images(images):
    # 이미지를 base64로 인코딩
    encoded_images = []
    for image in images:
        encoded_images.append(image)
    return encoded_images

def display_images(images):
    # 이미지를 Streamlit에 표시
    for image in images:
        st.image(image)

if uploaded_file and api_key:
    # PDF 파일에서 텍스트와 이미지 추출
    with st.spinner("PDF 파일에서 텍스트와 이미지를 추출하는 중..."):
        # text, images = extract_text_and_images_from_pdf(uploaded_file)
        # # BytesIO 객체를 사용하여 PDF 파일을 읽음
        # file = BytesIO(uploaded_file.read())
        # text, images = extract_text_and_images_from_pdf(file)

        # UploadedFile 객체를 임시 파일로 저장하여 해당 파일 경로를 사용
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # 업로드된 파일 내용을 임시 파일에 씀
            temp_file_path = temp_file.name  # 임시 파일의 경로를 얻음
        
        text, images = extract_text_and_images_from_pdf(temp_file_path)  # 임시 파일 경로를 사용하여 함수 호출

    # 텍스트를 청크로 분할
    chunks = text_to_chunks(text)
    
    # 텍스트 청크를 벡터 저장소에 저장
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vector_store = FAISS.from_texts(chunks, embeddings)

    # 대화형 체인 설정
    chain = load_qa_chain(OpenAI(openai_api_key=api_key), chain_type="stuff")

    # 사용자 인터페이스
    st.title("PDF 파일 챗봇")
    st.write("PDF 파일에서 추출한 텍스트와 이미지를 바탕으로 질문에 답변합니다.")

    if images:
        st.write("추출된 이미지:")
        display_images(images)
    
    user_question = st.text_input("질문을 입력하세요:")
    if user_question:
        with st.spinner("답변을 생성하는 중..."):
            docs = vector_store.similarity_search(user_question, k=5)
            answer = chain.run(input_documents=docs, question=user_question)
            st.write("답변:", answer)
else:
    st.write("PDF 파일을 업로드하고 OpenAI API 키를 입력하세요.")
