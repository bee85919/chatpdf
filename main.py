from dotenv import load_dotenv
load_dotenv()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Loader
loader = PyPDFLoader("luckyday.pdf")
pages = loader.load_and_split()


# Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 20,
    length_function = len,
    is_separator_regex = False,
)

texts = text_splitter.split_documents(pages)


# Embedding
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = OpenAIEmbeddings()


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


# Chroma
db = Chroma.from_documents(texts, embeddings_model)


# Generate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

question = "아내가 먹고 싶어하는 음식은 무엇이야?"
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
result = qa_chain({"query": question})

print(result)