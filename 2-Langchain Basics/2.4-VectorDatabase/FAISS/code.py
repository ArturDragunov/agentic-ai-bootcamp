#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv


# In[2]:


load_dotenv()


# In[3]:


import os
os.environ['HF_TOKEN']=os.getenv("HUGGINGFACEHUB_API_TOKEN")


# In[4]:


from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# In[5]:


embeddings.embed_query("hello AI")


# In[7]:


from sklearn.metrics.pairwise import cosine_similarity


# In[8]:


documents=["what is a capital of USA?",
           "Who is a president of USA?",
           "Who is a prime minister of India?"]


# In[9]:


my_query="Narendra modi is prime minister of india?"


# In[10]:


document_embedding=embeddings.embed_documents(documents)


# In[12]:


len(document_embedding) # each string got its own vector representation


# In[13]:


query_embedding=embeddings.embed_query(my_query) # almost the same as embed_documents but here input must be string, not list


# In[14]:


len(query_embedding)


# In[15]:


cosine_similarity([query_embedding],document_embedding) # you are comparing the cosine similarity between your query and other 3 strings. Higher means the angle between two vectors is smaller


# In[16]:


from sklearn.metrics.pairwise import euclidean_distances


# In[17]:


euclidean_distances([query_embedding], document_embedding) # it represents distances between two vectors. Here, sentence three is closer to the query (smaller euclidean distance)

#%%[markdown]
# Similarity Metrics Comparison
"""
| Metric | Formula | Range | Behavior | When to Use | Limitations | Notes |
|--------|---------|--------|----------|-------------|-------------|-------|
| **Cosine Similarity** | `cos(θ) = (x·y)/(‖x‖ ‖y‖)` | [-1, 1] | Focuses on angle only | High-dim sparse data, text analysis, when magnitude irrelevant | Ignores magnitude completely, undefined for zero vectors | Convert to distance: `1 - cosine_sim` |
| **L2 Distance (Euclidean)** | `√(∑(xᵢ - yᵢ)²)` | [0, ∞) | Focuses on **magnitude + direction** | Low-dim dense data, spatial problems, similar-scale features | Scale sensitive, curse of dimensionality, requires normalization | Convert to similarity: `1/(1 + distance)` |
| **Manhattan Distance (L1)** | `∑\|xᵢ - yᵢ\|` | [0, ∞) | Less sensitive to outliers | Robust to outliers, grid-like spaces, mixed data types | Still scale sensitive, less intuitive geometrically | More robust than L2 |
| **Jaccard Similarity** | `\|A ∩ B\| / \|A ∪ B\|` | [0, 1] | Set overlap proportion | Binary/categorical data, recommendation systems | Only for binary/set data, ignores frequency | Distance: `1 - jaccard` |
| **Hamming Distance** | `∑(xᵢ ≠ yᵢ)` | [0, n] | Count of differing positions | Binary strings, categorical data, error detection | Fixed-length vectors only, treats all differences equally | Normalized version: divide by n |
| **Pearson Correlation** | `cov(x,y)/(σₓσᵧ)` | [-1, 1] | Linear relationship strength | Time series, when linear correlation matters | Only captures linear relationships, sensitive to outliers | Distance: `1 - \|correlation\|` |

## Key Decision Framework

### By Data Type:
- **Continuous, similar scales** → Euclidean Distance
- **Continuous, different scales** → Cosine Similarity (after normalization)
- **High-dimensional sparse** → Cosine Similarity
- **Binary/categorical** → Jaccard or Hamming
- **Time series** → Pearson Correlation

### By Problem Context:
- **Magnitude matters** → Euclidean/Manhattan
- **Direction/pattern matters** → Cosine
- **Outlier robustness needed** → Manhattan
- **Set similarity** → Jaccard

### Preprocessing Requirements:
- **L1, L2**: Standardize features with different scales
- **Cosine**: Already scale-invariant
- **Jaccard, Hamming**: No preprocessing needed

FAISS is Facebook AI Similarity Search
"""
# In[18]:


import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


# In[19]:


index=faiss.IndexFlatL2(384) # inside index we will store our data. 384 is the embedding size (nr of feature representations)
# we always have to create index! We use euclidean distance index.
# What index does:
# Stores vectors: Your embeddings live inside the index
# Enables search: Provides fast nearest neighbor lookup
# Optimizes retrieval: Uses algorithms better than brute-force comparison

# L2 index 
# Flat: Brute-force exact search (no approximation)
# L2: Uses Euclidean distance
# 384: Must match your embedding size exactly


# Other index types:

# IndexFlatIP: Inner product (cosine similarity)
# IndexIVFFlat: Faster approximate search
# IndexHNSW: Graph-based fast search

# The index type determines both accuracy and speed of your similarity search.

# In[21]:


# creating a vector store!
vector_store=FAISS(
    embedding_function=embeddings, # here, hugging face
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


# In[22]:


vector_store.add_texts(["AI is future","AI is powerful","Dogs are cute"]) # they are automatically vectorized and added to the table


# In[23]:


vector_store.index_to_docstore_id


# In[28]:


results = vector_store.similarity_search("Tell me about AI", k=2) # it gives you 3 closest based on L2 distance texts.

#%%[markdown]
"""
| Feature               | `Flat`                | `IVF` (Inverted File Index)        | `HNSW` (Graph-based Index)          |
| --------------------- | --------------------- | ---------------------------------- | ----------------------------------- |
| Type of Search     | Exact                 | Approximate (cluster-based)        | Approximate (graph-based traversal) |
| Speed               | Slow (linear scan)    | Fast (search only in top clusters) | Very Fast (graph walk)              |


| Dataset Size              | Recommended Index                 |
| ------------------------- | --------------------------------- |
| UPTO 1L                     | `IndexFlatL2` or `IndexFlatIP`    |
| UPTO 1M                  | `IndexIVFFlat` or `IndexHNSWFlat` |
| > 1M                      | `IndexIVFPQ` or `IndexHNSWFlat`   |
"""

# In[30]:

# from uuid import uuid4
from langchain_core.documents import Document

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
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


# In[35]:


index=faiss.IndexFlatIP(384) # cosine similarity
vector_store=FAISS( # this object automatically does embedding and it adds the vector to the database
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


# In[36]:


vector_store.add_documents(documents=documents) # stored all 10 documents in database. if you run it twice, you will get same texts with two indexes (20 rows)


# In[37]:


vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2 #hyperparameter -> number of documents I want to retrieve

)


# In[38]:


vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    #k=2 #hyperparameter,
    filter={"source":{"$eq": "tweet"}} # we retrieve data only from tweet sourced information

)


# In[39]:


result=vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    #k=2 #hyperparameter,
    filter={"source":"news"}

)


# In[40]:


result[0].metadata


# In[41]:


result[0].page_content


# In[42]:


retriever=vector_store.as_retriever(search_kwargs={"k": 3}) # you can transform vector store as a retriever to use it inside RAG pipeline


# In[43]:


retriever.invoke("LangChain provides abstractions to make working with LLMs easy")


# In[ ]:


# inmemory(server)
# ondisk(server)
# cloud(yet to discuss)


# In[44]:


vector_store.save_local("today's class faiss index")


# In[45]:


new_vector_store=FAISS.load_local(
  "today's class faiss index",embeddings ,allow_dangerous_deserialization=True
)


# In[46]:


new_vector_store.similarity_search("langchain")


# In[47]:


from langchain_community.document_loaders import PyPDFLoader


# In[48]:


FILE_PATH=r"C:\Users\Artur Dragunov\Documents\GIT\agentic-ai-bootcamp\2-Langchain Basics\llama2-bf0a30209b224e26e31087559688ce81.pdf"


# In[49]:


loader=PyPDFLoader(FILE_PATH)


# In[50]:


len(loader.load())


# In[51]:


pages=loader.load() # it loads all pages


# In[52]:


pages = []
async for page in loader.alazy_load(): # load pages in parallel using asyncio
    pages.append(page)


# In[53]:


from langchain_text_splitters import RecursiveCharacterTextSplitter


# In[54]:


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,#hyperparameter
    chunk_overlap=50 #hyperparemeter
)


# In[55]:


split_docs = splitter.split_documents(pages)


# In[56]:


len(split_docs)


# In[57]:


index=faiss.IndexFlatIP(384)
vector_store=FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


# In[58]:


vector_store.add_documents(documents=split_docs) # added 615 chunks with cosine similarity index


# In[59]:


retriever=vector_store.as_retriever(
    search_kwargs={"k": 10} #hyperparameter
)


# In[60]:


retriever.invoke("what is llama model?")


# In[62]:


from langchain_openai import ChatOpenAI

model = ChatOpenAI(model = "gpt-4.1-mini-2025-04-14")


# In[ ]:


from langchain import hub
prompt = hub.pull("rlm/rag-prompt") # RAG prompt template


# In[66]:


import pprint
pprint.pprint(prompt.messages)


# [HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"), additional_kwargs={})]

# In[67]:


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# RunnablePassthrough() passes input data through unchanged
# it's essentially a "do nothing" operation that forwards whatever it receives.


# In[ ]:


# context(retriever),prompt(hub),model(openai),parser(langchain)


# In[68]:


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



# In[69]:


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


# In[70]:


rag_chain.invoke("what is llama model?")
