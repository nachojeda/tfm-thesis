# pip insatall langchain
# pip instal pypdf

pdf_path = "C:/Users/Nacho/Documents/MASTER/TFM/1-Pruebas_piloto/External data/BOE-034_Codigo_Civil_y_legislacion_complementaria.pdf"
# Load pdf with external info not seen during training of the LLM
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# pip install tiktoken
# Generate vector space represetnation with words from the external data
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# pip install faiss-cpu
# Load embeddings in vector database
from langchain.vectorstores import FAISS
db = FAISS.from_documents(pages, embeddings)

# q = "What is Link's traditional outfit color?"
# db.similarity_search(q)[0]

# Use information retrieval from embedding for answer
from langchain.chains import RetrievalQA
from langchain import OpenAI
llm = OpenAI()
chain = RetrievalQA.from_llm(llm=llm, retriever=db.as_retriever())
q = "What is Link's traditional outfit color?"
chain(q, return_only_outputs=True)