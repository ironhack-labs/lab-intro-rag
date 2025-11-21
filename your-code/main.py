# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="RsnCPbdkxYZd"
# <div style="text-align: center;">
#     <h1 style="color: #FF6347;">Self-Guided Lab: Retrieval-Augmented Generation (RAGs)</h1>
# </div>

# %% [markdown] id="tZp4BQAVxYZj"
# <div style="text-align: center;">
#     <img src="https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExZ3FsdzRveTBrenMxM3VnbDMwaTJxN2NnZm50aGFibXk1NzNnY2Q0MCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LR5ZBwZHv02lmpVoEU/giphy.gif" alt="NLP Gif" style="width: 300px; height: 150px; object-fit: cover; object-position: center;">
# </div>

# %% [markdown] id="gizk6HCYxYZo"
# <h1 style="color: #FF6347;">Data Storage & Retrieval</h1>
#

# %% [markdown] id="QW5UOI8ZxYZp"
# <h2 style="color: #FF8C00;">PyPDFLoader</h2>
#
# `PyPDFLoader` is a lightweight Python library designed to streamline the process of loading and parsing PDF documents for text processing tasks. It is particularly useful in Retrieval-Augmented Generation workflows where text extraction from PDFs is required.
#
# - **What Does PyPDFLoader Do?**
#   - Extracts text from PDF files, retaining formatting and layout.
#   - Simplifies the preprocessing of document-based datasets.
#   - Supports efficient and scalable loading of large PDF collections.
#
# - **Key Features:**
#   - Compatible with popular NLP libraries and frameworks.
#   - Handles multi-page PDFs and embedded images (e.g., OCR-compatible setups).
#   - Provides flexible configurations for structured text extraction.
#
# - **Use Cases:**
#   - Preparing PDF documents for retrieval-based systems in RAGs.
#   - Automating the text extraction pipeline for document analysis.
#   - Creating datasets from academic papers, technical manuals, and reports.
#

# %%
# %pip install langchain langchain_community pypdf
# %pip install termcolor langchain_openai langchain-huggingface sentence-transformers chromadb langchain_chroma tiktoken openai python-dotenv


# %% id="6heKZkQUxYZr"
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import warnings
warnings.filterwarnings('ignore')


# %% [markdown] id="sRS44B2XxYZs" vscode={"languageId": "plaintext"}
# <h3 style="color: #FF8C00;">Loading the Documents</h3>

# %% id="cuREtJRixYZt"
# File path for the document

file_path = "LAB/ai-for-everyone.pdf"

# %% [markdown] id="pz_8SOLxxYZt"
# <h3 style="color: #FF8C00;">Documents into pages</h3>
#
# The `PyPDFLoader` library allows efficient loading and splitting of PDF documents into smaller, manageable parts for NLP tasks.
#
# This functionality is particularly useful in workflows requiring granular text processing, such as Retrieval-Augmented Generation (RAG).
#

# %% id="_b5Z_45UxYZu" outputId="a600d69f-14fe-4492-f236-97261d6ff36c"
# Load and split the document
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
len(pages)

# %% [markdown] id="wt50NRQaxYZv"
# <h3 style="color: #FF8C00;">Pages into Chunks</h3>
#
#
# ####  RecursiveCharacterTextSplitter in LangChain
#
# The `RecursiveCharacterTextSplitter` is the **recommended splitter** in LangChain when you want to break down long documents into smaller, semantically meaningful chunks — especially useful in **RAG pipelines**, where clean context chunks lead to better LLM responses.
#
# ####  Parameters
#
# | Parameter       | Description                                                                 |
# |-----------------|-----------------------------------------------------------------------------|
# | `chunk_size`    | The **maximum number of characters** allowed in a chunk (e.g., `1000`).     |
# | `chunk_overlap` | The number of **overlapping characters** between consecutive chunks (e.g., `200`). This helps preserve context continuity. |
#
# ####  How it works
# `RecursiveCharacterTextSplitter` attempts to split the text **intelligently**, trying the following separators in order:
# 1. Paragraphs (`"\n\n"`)
# 2. Lines (`"\n"`)
# 3. Sentences or words (`" "`)
# 4. Individual characters (as a last resort)
#
# This makes it ideal for handling **natural language documents**, such as PDFs, articles, or long reports, without breaking sentences or paragraphs in awkward ways.
#
#

# %%
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(pages)

len(chunks)

# %% [markdown]
# ####  Alternative: CharacterTextSplitter
#
# `CharacterTextSplitter` is a simpler splitter that breaks text into chunks based **purely on character count**, without trying to preserve any natural language structure.
#
# ##### Example:
# ```python
# from langchain.text_splitter import CharacterTextSplitter
#
# text_splitter = CharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )
# ````
#
# This method is faster and more predictable but may split text in the middle of a sentence or paragraph, which can hurt performance in downstream tasks like retrieval or QA.
#
# ---
#
# #### Comparison Table
#
# | Feature                        | RecursiveCharacterTextSplitter | CharacterTextSplitter     |
# | ------------------------------ | ------------------------------ | ------------------------- |
# | Structure-aware splitting      |  Yes                          |  No                      |
# | Preserves sentence/paragraphs  |  Yes                          |  No                      |
# | Risk of splitting mid-sentence |  Minimal                     |  High                   |
# | Ideal for RAG/document QA      |  Highly recommended           |  Only if structured text |
# | Performance speed              |  Slightly slower             |  Faster                  |
#
# ---
#
# #### Recommendation
#
# Use `RecursiveCharacterTextSplitter` for most real-world document processing tasks, especially when building RAG pipelines or working with structured natural language content like PDFs or articles.

# %% [markdown]
# ## Best Practices for Choosing Chunk Size in RAG
#
# ### Best Practices for Chunk Size in RAG
#
# | Factor                      | Recommendation                                                                                                                                                                                          |
# | ---------------------------| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
# | **LLM context limit**       | Choose a chunk size that lets you retrieve multiple chunks **without exceeding the model’s token limit**. For example, GPT-4o supports 128k tokens, but with GPT-3.5 (16k) or GPT-4 (32k), keep it modest. |
# | **Chunk size (in characters)** | Typically: **500–1,000 characters** per chunk → ~75–200 tokens. This fits well for retrieval + prompt without context overflow.                                                                           |
# | **Chunk size (in tokens)**  | If using token-based splitter (e.g. `TokenTextSplitter`): aim for **100–300 tokens** per chunk.                                                                                                            |
# | **Chunk overlap**           | Use **overlap of 10–30%** (e.g., 100–300 characters or ~50 tokens) to preserve context across chunk boundaries and avoid cutting off important ideas mid-sentence.                                        |
# | **Document structure**      | Use **`RecursiveCharacterTextSplitter`** to preserve semantic boundaries (paragraphs, sentences) instead of arbitrary cuts.                                                                                |
# | **Task type**               | For **question answering**, smaller chunks (~500–800 chars) reduce noise.<br>For **summarization**, slightly larger chunks (~1000–1500) are OK.                                                          |
# | **Embedding model**         | Some models (e.g., `text-embedding-3-large`) can handle long input. But still, smaller chunks give **finer-grained retrieval**, which improves relevance.                                                  |
# | **Query type**              | If users ask **very specific questions**, small focused chunks are better. For broader queries, bigger chunks might help.                                                                                  |
#
#
# ### Rule of Thumb
#
# | Use Case                 | Chunk Size      | Overlap |
# | ------------------------| --------------- | ------- |
# | Factual Q&A              | 500–800 chars   | 100–200 |
# | Summarization            | 1000–1500 chars | 200–300 |
# | Technical documents      | 400–700 chars   | 100–200 |
# | Long reports/books       | 800–1200 chars  | 200–300 |
# | Small LLMs (≤16k tokens) | ≤800 chars      | 100–200 |
#
#
# ### Avoid
#
# - Chunks >2000 characters: risks context overflow.
# - No overlap: may lose key information between chunks.
#
#

# %% [markdown] id="Mg15RjVPxYZw"
# <h2 style="color: #FF8C00;">Embeddings</h2>
#
# Embeddings transform text into dense vector representations, capturing semantic meaning and contextual relationships. They are essential for efficient document retrieval and similarity analysis.
#
# - **What are OpenAI Embeddings?**
#   - Pre-trained embeddings like `text-embedding-3-large` generate high-quality vector representations for text.
#   - Encapsulate semantic relationships in the text, enabling robust NLP applications.
#
# - **Key Features of `text-embedding-3-large`:**
#   - Large-scale embedding model optimized for accuracy and versatility.
#   - Handles diverse NLP tasks, including retrieval, classification, and clustering.
#   - Ideal for applications with high-performance requirements.
#
# - **Benefits:**
#   - Reduces the need for extensive custom training.
#   - Provides state-of-the-art performance in retrieval-augmented systems.
#   - Compatible with RAGs to create powerful context-aware models.
#

# %% id="L0xDxElwxYZw"
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# %% id="_WRIo3_0xYZx" outputId="78bfbbf3-9d25-4e31-bdbc-3e932e6bbfec"
load_dotenv()

# %% id="MNZfTng5xYZz" outputId="db1a7c85-ef9f-447e-92cd-9d097e959847"
api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# %% [markdown] id="EsSA7RKvxYZz"
# <h2 style="color: #FF8C00;">ChromaDB</h2>
#
# ChromaDB is a versatile vector database designed for efficiently storing and retrieving embeddings. It integrates seamlessly with embedding models to enable high-performance similarity search and context-based retrieval.
#
# ### Workflow Overview:
# - **Step 1:** Generate embeddings using a pre-trained model (e.g., OpenAI's `text-embedding-3-large`).
# - **Step 2:** Store the embeddings in ChromaDB for efficient retrieval and similarity calculations.
# - **Step 3:** Use the stored embeddings to perform searches, matching, or context-based retrieval.
#
# ### Key Features of ChromaDB:
# - **Scalability:** Handles large-scale datasets with optimized indexing and search capabilities.
# - **Speed:** Provides fast and accurate retrieval of embeddings for real-time applications.
# - **Integration:** Supports integration with popular frameworks and libraries for embedding generation.

# %% id="brKe6wUgxYZ0"
from langchain.vectorstores import Chroma

# %% id="VkjHR-RkxYZ0" outputId="bc11bda9-f283-457a-f584-5a06b95c4dd9"
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db_LAB")
print("ChromaDB created with document embeddings.")

# %% [markdown] id="27OdN1IVxYZ1"
# <h1 style="color: #FF6347;">Retrieving Documents</h1>
#

# %% [markdown]
# ### Exercice1: Write a user question that someone might ask about your book’s topic or content.

# %% id="XiLv-TfrxYZ1"
user_question = "" # User question
retrieved_docs = db.similarity_search(user_question, k=10) # k is the number of documents to retrieve

# %% id="qgWsh50JxYZ1" outputId="c8640c5d-5955-471f-fdd2-37096f5f68c7"
# Display top results
for i, doc in enumerate(retrieved_docs[:3]): # Display top 3 results
    print(f"Document {i+1}:\n{doc.page_content[36:1000]}") # Display content


# %% [markdown] id="XuGK8gL6xYZ1"
# <h2 style="color: #FF8C00;">Preparing Content for GenAI</h2>

# %% id="2iB3lZqHxYZ2"
def _get_document_prompt(docs):
    prompt = "\n"
    for doc in docs:
        prompt += "\nContent:\n"
        prompt += doc.page_content + "\n\n"
    return prompt


# %% id="2okzmuADxYZ2" outputId="0aa6cdca-188d-40e0-f5b4-8888d3549ea4"
# Generate a formatted context from the retrieved documents
formatted_context = _get_document_prompt(retrieved_docs)
print("Context formatted for GPT model.")

# %% [markdown] id="qzIczQNTxYZ2"
# <h2 style="color: #FF8C00;">ChatBot Architecture</h2>

# %% [markdown]
# ### Exercice2: Write a prompt that is relevant and tailored to the content and style of your book.

# %% id="tqxVh9s3xYZ3" outputId="97cca95d-4ab3-44d8-a76c-5713aad387d8"
prompt = f"""


"""


# %% id="0mjkQJ_ZxYZ3"
import openai

# %% [markdown]
# ### Exercice3: Tune parameters like temperature, and penalties to control how creative, focused, or varied the model's responses are.

# %% id="ylypRWRlxYZ4"
# Set up GPT client and parameters
client = openai.OpenAI()
model_params = {
    'model': 'gpt-4o',
    'temperature': ,  # Increase creativity
    'max_tokens': ,  # Allow for longer responses
    'top_p': ,        # Use nucleus sampling
    'frequency_penalty': ,  # Reduce repetition
    'presence_penalty':    # Encourage new topics
}

# %% [markdown] id="C8e942xDxYZ4"
# <h1 style="color: #FF6347;">Response</h1>
#

# %% id="4eXZO4pIxYZ4"
messages = [{'role': 'user', 'content': prompt}]
completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)

# %% id="wLPAcchBxYZ5" outputId="976c7800-16ed-41fe-c4cf-58f60d3230d2"
answer = completion.choices[0].message.content
print(answer)

# %% [markdown] id="VXVNXPwLxYaT"
# <img src="https://miro.medium.com/v2/resize:fit:824/1*GK56xmDIWtNQAD_jnBIt2g.png" alt="NLP Gif" style="width: 500px">

# %% [markdown] id="ldybhlqKxYaT"
# <h2 style="color: #FF6347;">Cosine Similarity</h2>
#
# **Cosine similarity** is a metric used to measure the alignment or similarity between two vectors, calculated as the cosine of the angle between them. It is the **most common metric used in RAG pipelines** for vector retrieval.. It provides a scale from -1 to 1:
#
# - **-1**: Vectors are completely opposite.
# - **0**: Vectors are orthogonal (uncorrelated or unrelated).
# - **1**: Vectors are identical.
#

# %% [markdown] id="1c1I1TNhxYaT"
# <img src="https://storage.googleapis.com/lds-media/images/cosine-similarity-vectors.original.jpg" alt="NLP Gif" style="width: 700px">

# %% [markdown] id="EoEMdNgQxYaU"
# <h2 style="color: #FF6347;">Keyword Highlighting</h2>
#
# Highlighting important keywords helps users quickly understand the relevance of the retrieved text to their query.

# %% id="nCXL9Cz1xYaV"
from termcolor import colored


# %% [markdown] id="xwDyofY0xYaV"
# The `highlight_keywords` function is designed to highlight specific keywords within a given text. It replaces each keyword in the text with a highlighted version using the `colored` function from the `termcolor` library.
#

# %% id="9y3E0YWExYaV"
def highlight_keywords(text, keywords):
    for keyword in keywords:
        text = text.replace(keyword, colored(keyword, 'green', attrs=['bold']))
    return text


# %% [markdown]
# ### Exercice4: add your keywords

# %% id="i7SkWPpnxYaW" outputId="28e82563-edba-4b41-acad-ec27e5ba134f"
query_keywords = [] # add your keywords
for i, doc in enumerate(retrieved_docs[:1]):
    snippet = doc.page_content[:200]
    highlighted = highlight_keywords(snippet, query_keywords)
    print(f"Snippet {i+1}:\n{highlighted}\n{'-'*80}")

# %% [markdown] id="AhV_Jf_LxYaX"
# 1. `query_keywords` is a list of keywords to be highlighted.
# 2. The loop iterates over the first document in retrieved_docs.
# 3. For each document, a snippet of the first 200 characters is extracted.
# 4. The highlight_keywords function is called to highlight the keywords in the snippet.
# 5. The highlighted snippet is printed along with a separator line.

# %% [markdown] id="pBRKysAvxYaX"
# <h1 style="color: #FF6347;">Bonus</h1>

# %% [markdown] id="Qj25lCybxYaX"
# **Try loading one of your own PDF books and go through the steps again to explore how the pipeline works with your content**:
#
