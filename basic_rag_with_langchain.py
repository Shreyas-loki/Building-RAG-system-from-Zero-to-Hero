import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


#### INDEXING ####

# Load blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

#### RETREIVAL ####

vectorstore = Chroma.from_documents(documents=splits,
                                    embedding=AzureOpenAIEmbeddings(azure_deployment="ada0021_6",
                                                                    openai_api_version=AZURE_OPENAI_API_VERSION,
                                                                    chunk_size=1)
                                   )

# retreiver for fetching nearest documents
retriever = vectorstore.as_retriever()

llm = AzureChatOpenAI(
            openai_api_type="azure",
            openai_api_version=AZURE_OPENAI_API_VERSION,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name="gpt35turbo",
            temperature=0
        )

# Getting a RAG prompt posted on huggingface and using it as prompt template for our case
prompt_hub_rag = hub.pull("rlm/rag-prompt")
'''
Prompt looks like this:
------------------------
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}

Answer:
'''

#### GENERATION ####

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_hub_rag
    | llm
    | StrOutputParser()
)


# question = "What is Task Decomposition?"
question = input("Ask question: ")

print("Answer:", rag_chain.invoke(question))