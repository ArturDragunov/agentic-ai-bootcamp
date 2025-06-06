#!/usr/bin/env python
# coding: utf-8

# ### Embeddings  Techniques
# Convert Text Into Vectors

# In[1]:


import os
from dotenv import load_dotenv
load_dotenv() 
#https://python.langchain.com/docs/integrations/text_embedding/


# In[2]:


os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


# In[3]:


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# In[4]:


embeddings


# In[ ]:


text="This is a tutorial on OPENAI embeddings"
query_result=embeddings.embed_query(text) # convert text to a vector
query_result


# In[6]:


len(query_result) # any sentence will be converted on size 3072.


# In[7]:


text="This is a tutorial on OPENAI embeddings. My name is Krish"
query_result=embeddings.embed_query(text)
query_result


# In[8]:


len(query_result)


# In[9]:


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024) # control model dimensions -> smaller feature representation 
# (e.g. all your representation features are: royal, poor, palace, kingdom people etc.). with 3072 you will have 3072 such features, while with 1024, you will have 1024 features and the rest is not included
# -> less detailed embedding -> less nuances in the words.


# In[10]:


text="This is a tutorial on OPENAI embedding"
query_result=embeddings.embed_query(text)
len(query_result)


# In[11]:


from langchain_community.document_loaders import TextLoader

loader=TextLoader('speech.txt')
docs=loader.load()
docs


# In[12]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
final_documents=text_splitter.split_documents(docs)
final_documents


# In[13]:


final_documents[0].page_content


# In[ ]:


embeddings.embed_query(final_documents[0].page_content) # we convert text to vector. Then we will load it to database

