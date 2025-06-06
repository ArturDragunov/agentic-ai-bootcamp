#!/usr/bin/env python
# coding: utf-8

# ### Data Ingestion or Data Loader
# https://python.langchain.com/docs/integrations/document_loaders/

# In[1]:


### text loader
from langchain_community.document_loaders.text import TextLoader


# In[ ]:


loader=TextLoader('speech.txt')
loader


# In[ ]:


text_documents=loader.load()
text_documents # we can't parse it directly to LLM. We need it split it first


# In[ ]:


### Read a PDf file
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('syllabus.pdf')
docs=loader.load()
docs


# In[ ]:


## Web based loader - read from a website
from langchain_community.document_loaders import WebBaseLoader
import bs4
loader=WebBaseLoader(web_paths=("https://python.langchain.com/docs/integrations/document_loaders/","https://platform.openai.com/docs/pricing"),) # you can provide any amount of links
docs=loader.load()
docs


# In[ ]:


## Web based loader
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title","post-content","post-header") # Souptrainer is targeting specific class of elements from the webpage (html tags) to scrap
                     ))
                     )

doc=loader.load()
doc


# In[ ]:


#Arxiv
from langchain_community.document_loaders import ArxivLoader
docs = ArxivLoader(query="1706.03762", load_max_docs=2).load() # load arxiv articles. Take up to 2 documents which match
docs


# In[ ]:


len(docs)


# In[ ]:


from langchain_community.document_loaders import WikipediaLoader
docs = WikipediaLoader(query="Generative AI", load_max_docs=4).load()
len(docs)
print(docs)


# In[ ]:


from langchain_community.document_loaders import RecursiveUrlLoader

loader = RecursiveUrlLoader(
    "https://www.autoocista.cz/o-nas/",
)
print(loader.load())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




