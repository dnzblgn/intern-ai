---
title: Fastener Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Documentation for RAG System with Image Processing

## Overview
This system integrates **Retrieval-Augmented Generation (RAG)** with **image classification** to help users identify geometric objects and retrieve relevant fastener and manufacturing recommendations.

## 1- Image Processing and Feature Extraction
ResNet50 is modified to serve as a **feature extractor** rather than a classifier. The final classification layer is removed and the model to output a meaningful **feature vector (embedding)** instead of assigning a label. This ensures that the model captures **important patterns in the image** without making predictions.

To ensure consistent input processing, images are resized to **224×224 pixels**, converted into tensors, and normalized. These preprocessing steps help the model process images effectively and produce consistent embeddings.

Once an image is fed into ResNet50, it outputs a **high-dimensional feature vector** that represents the image's most significant characteristics. This embedding is then converted into a **NumPy array** for compatibility with further computations and storage. The following libraries are used in this process:
- **Torchvision (`torchvision.models`)**: Provides the pre-trained ResNet50 model.
- **Torch (`torch`)**: Used to process images as tensors and extract feature embeddings.
- **NumPy (`numpy`)**: Converts extracted embeddings from PyTorch tensors to NumPy arrays for easier storage and further processing.

## Storing and Retrieving Embeddings
The extracted embeddings are stored in a **Python dictionary** along with labels describing their geometric classification. Each entry consists of:
- The **embedding** (numerical vector representation of the image).
- The **label** describing the detected geometry ( "Flat or Sheet-Based", "Cylindrical", "Complex Multi Axis Geometry").

Instead of using a dedicated vector database, the embeddings are kept in-memory for quick access. 

## Image Similarity and Classification
When a new image is uploaded, the system extracts its embedding using the same process. This new embedding is compared to the stored reference embeddings using **cosine similarity**. The reference image with the highest similarity score is selected as the closest match, ensuring accurate geometric classification.

## 2- Document Processing and Retrieval
- **Extracting Text:** The system processes **Fastener_Types_Manual.docx** and **Manufacturing Expert Manual.docx** stored as `.docx` files. 

- **Chunking and Embedding Creation:** The text is extracted, divided into **small chunks** (1500 characters each with some overlap - if doesnt work try reducing chunk size) since an LLM can process only a limited number of tokens at a time and converted into embeddings using a **sentence-transformers model**. 
- **Vector database:** These embeddings are stored in **FAISS**, a fast similarity search database.
- **Creating Embeddings:** Text chunks are converted into embeddings using **`HuggingFaceEmbeddings` (`BAAI/bge-base-en-v1.5`)**.
- **Vector Search:** These embeddings are stored in **FAISS (Facebook AI Similarity Search)** for fast lookups.
- **Cross-Encoder reranker:** To improve accuracy, we use a **Cross-Encoder reranker** (`sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`). It re-ranks these retrieved chunks (for example you get 5 chunks from vector search), to ensure that only the most useful and contextually relevant chunks are passed to the LLM for response generation..
- **Matching Queries:** When a user asks a question, the system creates an embedding of the query and finds the most relevant text chunks using **cosine similarity** in FAISS.
- Only chunks with a similarity score above **0.5** are considered relevant.


### 3- Query Validation & Response Generation
Finding relevant text isn’t enough—we need to make sure it actually answers the user’s question.

- **LLM Response Generation:** We use **Falcon-40B-Instruct** via `HuggingFaceEndpoint` to generate answers.
- **Checking Relevance:** The system checks if the retrieved text matches the question using `sentence-transformers`.
- **Fallback Handling:** If the system finds no useful information, it ignores irrelevant text and gives a fallback response.

This makes sure the chatbot provides accurate and meaningful answers, not just random AI-generated text.  


### 4- **Preventing Hallucinations**
#### Initial Retrieval from FAISS (Based on Cosine Similarity)**
- The system first retrieves the top **k** chunks from FAISS based on **cosine similarity**.
- If **no chunks meet the retrieval threshold** (e.g., `0.5`), it means **no relevant document was found**.
- In this case, the system **does not proceed further** and will return:
  ```
  "I couldn't find any relevant information."
  ```
####  What If FAISS Returns Low-Scoring Chunks?**
- If FAISS still **returns chunks** (but they have **low similarity scores**), the system applies **semantic validation** using a second, **lower threshold** (e.g., `0.3`).
- This acts as a **double-filtering mechanism**:
  - If even the **best retrieved chunks** fail this validation threshold, they are discarded.
  - The system **avoids generating an incorrect response** and instead returns a fallback answer.

#### Does FAISS Always Return Chunks?**
- FAISS will **always try to return the `k` closest chunks**, even if they are **not relevant**.
- However, if their **cosine similarity scores are too low**, they are **removed before being passed to the LLM**.

This step prevents **hallucinations**, where AI might generate answers that sound correct but have no real basis.

The **LLM uses both the user’s question and retrieved documents** to generate a final response. This way, the answer is based on real information instead of making things up.
