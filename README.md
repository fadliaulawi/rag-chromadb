# Retrieval Augmented Generation on Biomedical Data

This repository contains code for a paper on biomedical RAG, to be submitted to PRICAI 2024 in collaboration with BRIN.

![RAG](assets/embedding-based_retrieval.png)

## Requirements
- Python 3.10

## Development
1. Clone the repository
    ```bash
    git clone https://github.com/KalbeDigitalLab/rag-kak-pricai
    ```
2. Create a virtual environment and activate it
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    or on Windows
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. Install the dependencies
   ```bash
    pip install -r requirements.txt
    ```
4. Create a .env file in the root directory and add the following variables
    ```bash
    MONGODB_URI = "string"
    OPENAI_API_KEY = "string"
    QDRANT_API_KEY = "string"
    QDRANT_API_URL = "string"
    ```
5. Set the environment in the jupyter notebook
6. Run the jupyter notebook