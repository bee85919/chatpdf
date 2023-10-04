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