#!/usr/bin/env python
# coding: utf-8

# #### Text Splitting from Documents- RecursiveCharacter Text Splitters
# This text splitter is the recommended one for generic text. It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""]. This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.
# 
# - How the text is split: by list of characters.
# - How the chunk size is measured: by number of characters.
# - ["\n\n", "\n", " ", ""] is a list of characters we split
# - Splitting is not strict. If you specify 1000 as a chunk size, splitter WILL TRY to keep the chunks within the size BUT the split is based on the list of splitting characters

# In[ ]:


## Reading a PDf File
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('syllabus.pdf')
docs=loader.load()
docs


# In[ ]:


from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100) # 1000 number of characters for the split. # 100 overlap between 2 neighboring chunks
final_documents=text_splitter.split_documents(docs)
final_documents


# In[ ]:


final_documents[1]


# In[ ]:


final_documents[2]


# In[ ]:


final_documents[3]


# In[ ]:


print(final_documents[0])
print(final_documents[1])


# In[ ]:


## Text Loader

from langchain_community.document_loaders import TextLoader

loader=TextLoader('speech.txt')
docs=loader.load()
docs


# In[ ]:


speech=""
with open("speech.txt") as f:
    speech=f.read()


text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
text=text_splitter.create_documents([speech])
print(text[0])
print(text[1])


# Usual chunk size is 1000 and overlap is 200

# #### How to split by character-Character Text Splitter
# This is the simplest method. This splits based on a given character sequence, which defaults to "\n\n". Chunk length is measured by number of characters.
# 
# 1. How the text is split: by single character separator.
# 2. How the chunk size is measured: by number of characters.
# 

# In[ ]:


from langchain_text_splitters import CharacterTextSplitter
text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=100,chunk_overlap=20)
text_splitter.split_documents(docs) 


# In[ ]:


speech=""
with open("speech.txt") as f:
    speech=f.read()


text_splitter=CharacterTextSplitter(chunk_size=100,chunk_overlap=20)
text=text_splitter.create_documents([speech])
print(text[0])
print(text[1])


# ##### How to split by HTML header
# HTMLHeaderTextSplitter is a "structure-aware" chunker that splits text at the HTML element level and adds metadata for each header "relevant" to any given chunk. It can return chunks element by element or combine elements with the same metadata, with the objectives of (a) keeping related text grouped (more or less) semantically and (b) preserving context-rich information encoded in document structures. It can be used with other text splitters as part of a chunking pipeline.
# 

# In[ ]:


from langchain_text_splitters import HTMLHeaderTextSplitter

html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

headers_to_split_on=[ # list of tags I want to split on
    ("h1","Header 1"),
    ("h2","Header 2"),
    ("h3","Header 3")
]

html_splitter=HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits=html_splitter.split_text(html_string)
html_header_splits


# #### How to split JSON data
# This json splitter splits json data while allowing control over chunk sizes. It traverses json data depth first and builds smaller json chunks. It attempts to keep nested json objects whole but will split them if needed to keep chunks between a min_chunk_size and the max_chunk_size.
# 
# If the value is not a nested json, but rather a very large string the string will not be split. If you need a hard cap on the chunk size consider composing this with a Recursive Text splitter on those chunks. There is an optional pre-processing step to split lists, by first converting them to json (dict) and then splitting them as such.
# 
# - How the text is split: json value.
# - How the chunk size is measured: by number of characters.

# In[11]:


import json
import requests

json_data=requests.get("https://api.smith.langchain.com/openapi.json").json()
from langchain_text_splitters import RecursiveJsonSplitter
json_splitter = RecursiveJsonSplitter(max_chunk_size=1000)
json_splitter.split_json(json_data)


# In[ ]:


from langchain_text_splitters import RecursiveJsonSplitter
json_splitter=RecursiveJsonSplitter(max_chunk_size=300)
json_chunks=json_splitter.split_json(json_data)
json_chunks

