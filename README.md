This is the basic RAG implementation using Unstructured.io as parser, Pathway as vectorstore, Cohere Rerank3 for reranking, Google's text-embedding-004 for embedding and Gemini-1.5-Pro as LLM.

## Installation
1. Clone the repository
2. Install the requirements
3. Add the `credentials.json` file and `.env` file in the root directory
4. Run the ragServer.py file to run the pathway server
5. Run the rag.py file to run the client and get the result for your query

## .env file
```
UNSTRUCTURED_API_KEY = ""
COHERE_API_KEY = ""
GOOGLE_API_KEY = ""
```