__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import openai
import base64
import tempfile
import uuid
import re
import io
import pytesseract
import shutil
import platform
import nltk
nltk.download('punkt')

from PIL import Image
#import PIL.Image as pil
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
from langchain.docstore.document import Document

# Streamlit 페이지 설정
st.set_page_config(page_title="PDF 파일 챗봇", page_icon=":robot:")

# OS 세부 정보 출력
os_info = platform.system()
node_name = platform.node()
os_release = platform.release()
os_version = platform.version()
machine = platform.machine()
processor = platform.processor()
python_version = platform.python_version()  # Python 버전 추가

st.write("### 현재 시스템 정보")
st.write(f"**System**: {os_info}")
st.write(f"**Node Name**: {node_name}")
st.write(f"**Release**: {os_release}")
st.write(f"**Version**: {os_version}")
st.write(f"**Machine**: {machine}")
st.write(f"**Processor**: {processor}")
st.write(f"**Python Version**: {python_version}")  # Python 버전 출력

# 사이드바에서 파일 업로드 및 OpenAI API 키 입력
st.sidebar.title("PDF 파일 챗봇")
uploaded_file = st.sidebar.file_uploader("PDF 파일 업로드", type=["pdf"])
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# OpenAI API 설정
if api_key:
    openai.api_key = api_key
    
# tmp 폴더 안의 파일 지우기
def clear_tmp_directory(directory="/tmp"):
    """
    지정된 디렉토리의 모든 파일과 폴더를 삭제하는 함수.
    기본값으로 /tmp 디렉토리를 사용함.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # 파일 또는 심볼릭 링크 삭제
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # 디렉토리와 그 안의 모든 내용 삭제
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
                
def extract_pdf_elements(path, fname):
    """
    PDF 파일에서 이미지, 테이블, 그리고 텍스트 조각을 추출합니다.
    path: 이미지(.jpg)를 저장할 파일 경로
    fname: 파일 이름
    """
    # 이미지 출력 경로 설정
    # image_output_dir = os.path.join(path, "extracted_images")
    # os.makedirs(image_output_dir, exist_ok=True)  # 디렉토리가 없으면 생성]
    
    return partition_pdf(
        filename=os.path.join(path, fname),
        strategy="hi_res",
        extract_images_in_pdf=True,  # PDF 내 이미지 추출 활성화
        extract_image_block_types=["Image", "Table"],
        chunking_strategy="by_title",  # 제목별로 텍스트 조각화
        max_characters=4000,  # 최대 문자 수
        new_after_n_chars=3800,  # 이 문자 수 이후에 새로운 조각 생성
        combine_text_under_n_chars=2000,  # 이 문자 수 이하의 텍스트는 결합
        extract_image_block_to_payload=True,
        extract_image_block_output_dir=path,  # 이미지 출력 디렉토리 경로
    )
    
# 이미지 경로를 새로 이동
def move_images_to_target_dir(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for img_file in os.listdir(source_dir):
        full_file_name = os.path.join(source_dir, img_file)
        if os.path.isfile(full_file_name):
            shutil.move(full_file_name, target_dir)
            
def move_images_to_target_dir(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for img_file in os.listdir(source_dir):
        full_file_name = os.path.join(source_dir, img_file)
        
        # 동일한 파일 이름이 이미 대상 경로에 존재하는지 확인
        destination_file = os.path.join(target_dir, img_file)
        if os.path.isfile(full_file_name):
            if os.path.exists(destination_file):
                # 파일 이름이 충돌하는 경우, 새로운 이름을 생성
                base, ext = os.path.splitext(img_file)
                new_file_name = f"{base}_{uuid.uuid4().hex[:8]}{ext}"
                destination_file = os.path.join(target_dir, new_file_name)
            
            shutil.move(full_file_name, destination_file)
def categorize_elements(raw_pdf_elements):
    """
    PDF에서 추출된 요소를 테이블과 텍스트로 분류합니다.
    raw_pdf_elements: unstructured.documents.elements의 리스트
    """
    tables = []  # 테이블 저장 리스트
    texts = []  # 텍스트 저장 리스트
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))  # 테이블 요소 추가
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))  # 텍스트 요소 추가
    return texts, tables
    
def generate_text_summaries(texts, tables, summarize_texts=False):
    """
    텍스트 요소 요약
    texts: 문자열 리스트
    tables: 문자열 리스트
    summarize_texts: 텍스트 요약 여부를 결정. True/False
    """
    # 프롬프트 설정
    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
    These summaries will be embedded and used to retrieve the raw text or table elements. \
    Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_template(prompt_text)
    # 텍스트 요약 체인
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    # 요약을 위한 빈 리스트 초기화
    text_summaries = []
    table_summaries = []
    # 제공된 텍스트에 대해 요약이 요청되었을 경우 적용
    if texts and summarize_texts:
        text_summaries = summarize_chain.batch(texts, {"max_concurrency": 5})
    elif texts:
        text_summaries = texts
    # 제공된 테이블에 적용
    if tables:
        table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})
    return text_summaries, table_summaries
    
def encode_image(image_path):
    # 이미지 파일을 base64 문자열로 인코딩합니다.
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
def image_summarize(img_base64, prompt):
    # 이미지 요약을 생성합니다.
    chat = ChatOpenAI(model="gpt-4", max_tokens=2048)
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content
    
def generate_img_summaries(path):
    """
    이미지에 대한 요약과 base64 인코딩된 문자열을 생성합니다.
    path: Unstructured에 의해 추출된 .jpg 파일 목록의 경로
    """
    img_base64_list = []
    image_summaries = []
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Give a concise summary of the image that is well optimized for retrieval."""
    for img_file in sorted(os.listdir(path)):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
    return img_base64_list, image_summaries
    
def create_multi_vector_retriever(
    vectorstore, text_summaries, texts, table_summaries, tables, image_summaries, images
):
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    return retriever
    
def add_documents(retriever, doc_summaries, doc_contents):
    id_key = "doc_id"
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [Document(page_content=s, metadata={id_key: doc_ids[i]}) for i, s in enumerate(doc_summaries)]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))
    
def plt_img_base64(img_base64):
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    display(HTML(image_html))
    
def looks_like_base64(sb):
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None
    
def is_image_data(b64data):
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False
        
def resize_base64_image(base64_string, size=(128, 128)):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    resized_img = img.resize(size, Image.LANCZOS)
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
def split_image_text_types(docs):
    b64_images = []
    texts = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}
    
def img_prompt_func(data_dict):
    """
    컨텍스트를 단일 문자열로 결합
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []
    # 이미지가 있으면 메시지에 추가
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    # 분석을 위한 텍스트 추가
    text_message = {
        "type": "text",
        "text": (
            "You are a pdf analyst who analyzes the uploaded PDF.\n"
            "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide appropriate answers to your questions. Answer in Korean.\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]
    
def multi_modal_rag_chain(retriever):
    """
    멀티모달 RAG 체인
    """
    # 멀티모달 LLM
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=2048)
    # RAG 파이프라인
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )
    return chain
    
if uploaded_file and api_key:
    #Clean /tmp Directory
    #os.deleteDir("/tmp") //
    # 애플리케이션 시작 시 /tmp 디렉토리 정리
    # clear_tmp_directory()
    # clear_tmp_directory()

    # PDF 파일에서 텍스트와 이미지 추출
    with st.spinner("PDF 파일에서 텍스트와 이미지를 추출하는 중..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            # temp_file_path = temp_file.name 파일과 경로 이름 헷갈리지 않고 코드 작성하기
            temp_file_name = os.path.basename(temp_file_path)  # 업로드된 파일 이름 저장
            temp_file_path = os.path.basepath(temp_file_path) # 위치 추출
            
    st.write(f"fname: {fname}")
    st.write(f"temp_file_path: {temp_file_path}")
    st.write(f"os.path.dirname(temp_file_path): {os.path.dirname(temp_file_path)}")
    tmp_files = os.listdir(os.path.dirname(temp_file_path))
    st.write(f"tem_files: {tmp_files}")
    
    
    # PDF 파일의 요소들을 추출하고 이미지가 저장된 경로를 반환
    raw_pdf_elements = extract_pdf_elements(os.path.dirname(temp_file_path), fname)
    #extract_path = os.path.dirname(temp_file_path) + "/extracted_images"
    #st.write(f"extract_path: {extract_path}")
    #extract_files = os.listdir(extract_path)
    #st.write(f"tem_files: {extract_files}")
    
    st.write(f"raw_pdf_elements: {raw_pdf_elements}")
    img = Image.open(temp_file_path)
    img
    # `image_output_dir`은 `extract_pdf_elements` 함수에서 지정한 이미지 저장 경로입니다.
    #st.wrtie(f"temp_file_path: {temp_file_path}")
    #st.write(f"extracted_images: {extracted_images}")
    
    # image_output_dir = os.path.join(temp_file_path, "extracted_images")
    # st.wrtie(f"image_output_dir: {image_output_dir}")
    
    # 추출된 이미지 경로 확인
    
    #if os.path.exists(image_output_dir):
    #    extracted_images = os.listdir(image_output_dir)
    #    st.write(f"Images in 'extracted_images' directory:")
    #    st.write(extracted_images)
    #    
    #    # 이미지를 표시하려면
    #    for img_file in extracted_images:
    #        img_path = os.path.join(image_output_dir, img_file)
    #        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
    #            image = Image.open(img_path)
    #            st.image(image, caption=img_file)
    #else:
    #    st.write("이미지가 존재하지 않습니다.")
    
    
    # 추출된 이미지들을 /tmp 디렉토리로 이동
    # target_directory = "/tmp"
    # move_images_to_target_dir(image_output_dir, target_directory)
    
    texts, tables = categorize_elements(raw_pdf_elements)
    # 이미지 저장 경로 확인 및 복사
    #base_directory = "/tmp"  # 또는 "/var/tmp" 등 필요한 기본 경로로 변경 가능
    #source_directory = find_image_directory(base_directory)
    #if source_directory:
    #    copy_images_to_tmp(source_directory)
    #    print(f"Images copied from {source_directory} to /tmp")
    #else:
    #    print("No images found in the specified directory.")
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
    # 벡터 저장소 생성
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
    # 텍스트, 테이블, 이미지 추가
    if text_summaries:
        add_documents(retriever_multi_vector_img, text_summaries, texts)
    if table_summaries:
        add_documents(retriever_multi_vector_img, table_summaries, tables)
    if image_summaries:
        add_documents(retriever_multi_vector_img, image_summaries, img_base64_list)
    # RAG 체인 생성
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
    # 사용자 인터페이스
    st.title("PDF 파일 챗봇")
    st.write("PDF 파일에서 추출한 텍스트와 이미지를 바탕으로 질문에 답변합니다.")
    user_question = st.text_input("질문을 입력하세요:")
    if user_question:
        with st.spinner("답변을 생성하는 중..."):
            docs = retriever_multi_vector_img.vectorstore.similarity_search(user_question, k=5)
            answer = chain_multimodal_rag.invoke("question")
            st.write("답변:", answer)
            # 질문에 이미지 요청이 있는지 확인
            if "이미지" in user_question or "사진" in user_question or "차트" in user_question:
                for doc in docs:
                    if is_image_data(doc.page_content):
                        plt_img_base64(doc.page_content)
                        break
else:
    st.write("PDF 파일을 업로드하고 OpenAI API 키를 입력하세요.")
