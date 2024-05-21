# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import params
import ssl, smtplib
import os

# Step 1: Load
# loaders = [
#  WebBaseLoader("https://en.wikipedia.org/wiki/AT%26T"),
#  WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")
# ]
# data = []
# for loader in loaders:
#     data.extend(loader.load())

ca = certifi.where()

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")

loader.requests_kwargs = {'verify':False}

data = loader.load()

# print(data)

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                               "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)

for doc in docs:
    print(doc)
    print('---')

# print(docs)
print('Split into ' + str(len(docs)) + ' docs')

# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

# for embedding in embeddings:
#     print(embedding)

# Step 4: Store
# Initialize MongoDB python client

print(params.mongodb_conn_string)

client = MongoClient(params.test,server_api=ServerApi('1'))
collection = client[params.db_name][params.collection_name]

# Reset w/out deleting the Search Index 
collection.delete_many({})

# Disable SSL verification


# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=params.index_name
)