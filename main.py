__import__('pysqlite3')
import sys
sys.modules['sqltie3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv

# load .env
load_dotenv()


import streamlit as st
import tempfile
import os


# title
st.title("ChatPDF")
st.write("---")


# upload
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type="pdf")
st.write("---")


def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages


# after upload
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)


    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)



    # Embedding
    embeddings_model = OpenAIEmbeddings()


    # Chroma
    db = Chroma.from_documents(texts, embeddings_model)


    # Generate Q&A
    st.header("PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요.")
    
    if st.button("질문하기"):
        with st.spinner("답변 생성중..."):        
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result["result"])