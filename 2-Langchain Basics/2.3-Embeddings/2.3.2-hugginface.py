#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from dotenv import load_dotenv
load_dotenv()  #load all the environment variables

#https://python.langchain.com/docs/integrations/text_embedding/ # different embedding techniques


# In[3]:


os.environ['HF_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")


# #### Sentence Transformers on Hugging Face
# Hugging Face sentence-transformers is a Python framework for state-of-the-art sentence, text and image embeddings. One of the embedding models is used in the HuggingFaceEmbeddings class. We have also added an alias for SentenceTransformerEmbeddings for users who are more familiar with directly using that package.

# In[ ]:


from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # model is downloaded and will be embedding locally


# In[5]:


embeddings


# In[6]:


text="this is atest documents"
query_result=embeddings.embed_query(text)
query_result


# In[7]:


len(query_result) # there is only 384 feature representations. Very poor model


# In[10]:


from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# In[11]:


vectors=embeddings.embed_query("Hello, world!")
len(vectors)

