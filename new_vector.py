# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas

# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import params
import ssl, smtplib
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# Step 1: Load
loaders = [
 WebBaseLoader("https://en.wikipedia.org/wiki/AT%26T"),
 WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")
]


data = []
for loader in loaders:
    loader.requests_kwargs = {'verify':False}
    data.extend(loader.load())

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                               "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)

for doc in docs:
    print(doc)
    print('---')

print('Split into ' + str(len(docs)) + ' docs')

uri = "mongodb+srv://michael:TYoM5DIbnTvJryFH@cluster-mackah666.gchi9bo.mongodb.net/?retryWrites=true"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
# Send a ping to confirm a successful connection
try:
    dump = client.list_database_names()
    print(dump)
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = OpenAIEmbeddings(openai_api_key=params.openai_api_key)

for embedding in embeddings:
    print(embedding)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Reset w/out deleting the Search Index 
collection.delete_many({})

# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
# docsearch = MongoDBAtlasVectorSearch.from_documents(
#     docs, embeddings, collection=collection, index_name=params.index_name
# )



# insert the documents in MongoDB Atlas Vector Search
x = MongoDBAtlasVectorSearch.from_documents(
documents=docs, embedding=OpenAIEmbeddings(open_api_key=params.openai_api_key), collection=collection, index_name=params.index_name
)