#!/usr/bin/env python
# coding: utf-8

# In[10]:
from dotenv import load_dotenv
load_dotenv()

# In[2]:

import os
os.environ['HF_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")

# In[5]:

from langchain_huggingface import HuggingFaceEmbeddings
# This code works for both first run and subsequent runs -> it first caches code and then in the future it checks and loads the model
embeddings = HuggingFaceEmbeddings(
  model_name="all-MiniLM-L6-v2",
  cache_folder="../../../hugging_face_embedding",
  model_kwargs={'device': 'cpu'}  # or 'cuda' if you have GPU
)

# In[4]:
embeddings.embed_query("hello AI")

# In[6]:
len(embeddings.embed_query("hello AI"))
# reload your jupyter

# In[11]:

from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # it takes env key directly from environment variables


# In[12]:


len(embeddings.embed_query("Hello AI"))


# In[13]:


from pinecone import Pinecone


# In[16]:


import os
pinecone_api_key=os.getenv("PINECONE_API_KEY")


# In[17]:


pc=Pinecone(api_key=pinecone_api_key)


# In[18]:


from pinecone import ServerlessSpec
#Serverless: Server will be Managed by the cloud provider


# In[19]:


index_name="agenticbatch2"


# In[ ]:


pc.has_index(index_name) # no index inside


# In[22]:


#creating an index
if not pc.has_index(index_name):
    pc.create_index( # right now we use flat index
    name=index_name,
    dimension=768, # we use google gemini model, so we use 768 as dimension
    metric="cosine",
    spec=ServerlessSpec(cloud="aws",region="us-east-1")    
)


# In[23]:


pc.has_index(index_name) # index is added now
# index is added: https://app.pinecone.io/organizations/-ORjEfC6X56LCBuIAroE/projects/060ef57d-4171-4b0e-9656-19818cbf0419/indexes


# In[24]:


#loading the index
index=pc.Index(index_name)


# In[25]:


from langchain_pinecone import PineconeVectorStore # now we can store the data under the index


# In[26]:


vector_store=PineconeVectorStore(index=index,embedding=embeddings) # we define a vector store under this specific index and using defined embedding. It's important to use
# correct embedding -> the one which was assumed during index creation. Otherwise, dimensions won't match.


# In[27]:


results = vector_store.similarity_search("what is a langchain?") # currently vectore store is empty


# In[28]:


results


# In[29]:


from uuid import uuid4
from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},#additional info
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
)

document_3 = Document(
    page_content="Building an exciting new project with LangChain - come check it out!",
    metadata={"source": "tweet"},
)

document_4 = Document(
    page_content="Robbers broke into the city bank and stole $1 million in cash.",
    metadata={"source": "news"},
)

document_5 = Document(
    page_content="Wow! That was an amazing movie. I can't wait to see it again.",
    metadata={"source": "tweet"},
)

document_6 = Document(
    page_content="Is the new iPhone worth the price? Read this review to find out.",
    metadata={"source": "website"},
)

document_7 = Document(
    page_content="The top 10 soccer players in the world right now.",
    metadata={"source": "website"},
)

document_8 = Document(
    page_content="LangGraph is the best framework for building stateful, agentic applications!",
    metadata={"source": "tweet"},
)

document_9 = Document(
    page_content="The stock market is down 500 points today due to fears of a recession.",
    metadata={"source": "news"},
)

document_10 = Document(
    page_content="I have a bad feeling I am going to get deleted :(",
    metadata={"source": "tweet"},
)


# In[30]:
documents = [
    document_1,
    document_2,
    document_3,
    document_4,
    document_5,
    document_6,
    document_7,
    document_8,
    document_9,
    document_10,
]


# In[31]:

documents


# In[32]:


len(documents)


# In[33]:


range(len(documents))


# In[34]:

for _ in range(len(documents)):
    print(_)
    print(str(uuid4()))



# In[35]:


#universal indentification number
uuids = [str(uuid4()) for _ in range(len(documents))] # -> it's automatically generated with FAISS


# In[36]:

uuids


# In[37]:


vector_store.add_documents(documents=documents, ids=uuids) 
# we add documents to vector_store with defined universal indentifaction number


# In[38]:


results = vector_store.similarity_search("what langchain provides to us?",k=1)


# In[39]:


results


# In[42]:


results = vector_store.similarity_search("what langchain provides to us?",k=2,filter={"source": "tweet"})


# In[43]:


results


# In[47]:


retriever=vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":3,"score_threshold": 0.7} #hyperparameter -> similarity level based on cos similarity + k closest matches
)


# In[48]:


retriever.invoke("langchain")


# In[49]:


retriever.invoke("google")


# In[50]:


from langchain_google_genai import ChatGoogleGenerativeAI
model=ChatGoogleGenerativeAI(model='gemini-1.5-flash')


# In[ ]:


from langchain import hub
prompt = hub.pull("rlm/rag-prompt") # we use rag-prompt


# In[52]:


import pprint
pprint.pprint(prompt.messages)


#%%[markdown]
"""
[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})]
"""

# In[53]:


from langchain_core.prompts import PromptTemplate


# In[54]:


prompt=PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
    input_variables=['context', 'question']
)


# In[55]:


prompt


# In[56]:


prompt.invoke({"question":"what is a langchain?","context":"langchain is very super framework for LLM."}) # question and context are needed parameters as langchain picks and submits them into Prompt

#%%[markdown]
# StringPromptValue(text="You are an assistant for question-answering tasks.
#  Use the following pieces of retrieved context to answer the question.
#  If you don't know the answer, just say that you don't know.
#  Use three sentences maximum and keep the answer concise.
# \nQuestion: what is a langchain? \nContext: langchain is very super framework for LLM. \nAnswer:")

# In[57]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# In[58]:


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# In[59]:


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# In[60]:


rag_chain.invoke("what is llama model?")


# # second assisgnment is: take a multiple pdf with text,image,table
# 1. fetch the data from pdf
# 2. at lesat there should be 200 pages
# 3. if chunking(use the sementic chunking technique) required do chunking and then embedding
# 4. store it inside the vector database(use any of them 1. mongodb 2. astradb 3. opensearch 4.milvus) ## i have not discuss then you need to explore
# 5. create a index with all three index machnism(Flat, HNSW, IVF) ## i have not discuss then you need to explore
# 6. create a retriever pipeline
# 7. check the retriever time(which one is fastet)
# 8. print the accuray score of every similarity search
# 9. perform the reranking either using BM25 or MMR ## i have not discuss then you need to explore
# 10. then write a prompt template
# 11. generte a oputput through llm
# 12. render that output over the DOCx ## i have not discuss then you need to explore
# as a additional tip: you can follow rag playlist from my youtube
# 
# after completing it keep it on your github and share that link on my  mail id:
# snshrivas3365@gmail.com
# 
# and share the assignment in your community chat as well by tagging krish and sunny
# 
# deadline is: till firday 9PM
#    
