# AC215 Final Project - README

#### Group Name
ClosedAI

#### Team Members
Nina Mao, Yunxiao Tang, Jiayi Xu, Xinjie Yi

#### Project
In this project, we aim to develop an AI-powered mental healing application. The app will feature advanced conversational technology to engage in meaningful dialogue with individuals experiencing negative psychological states. Powered by fine-tuned GEMINI and RAG models, the application is designed to offer specialized mental healing support. Users can interact with the app through natural conversations, where it draws from a wealth of expert psychology literature to provide professional, evidence-based mental health guidance. Whether users are dealing with stress, anxiety, or other emotional challenges, the app offers personalized therapeutic advice, helping them navigate difficult emotions and promote mental well-being.

## Milestone 2

In this milestone, we have completed the components for fine-tuning the GEMINI model and developed the RAG model for enhanced retrieval capabilities.

#### Project Organization

```
├── README.md
├── LICENSE
├── .gitignore
├── data
└── src
    ├── finetune_data
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── prepare_data.py
    │   ├── Pipfile
    │   └── Pipfile.lock
    └── finetune_model
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── finetune.py
    │   ├── Pipfile
    │   └── Pipfile.lock
    └── rag_data_pipeline
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── dataloader.py
    │   ├── preprocess_rag.py
    │   ├── Pipfile
    │   └── Pipfile.lock
    └── rag_model
    │   ├── docker-shell.sh
    │   ├── Dockerfile
    │   ├── model.py  
    │   ├── Pipfile
    │   └── Pipfile.lock
    ├── docker-compose.yml
    ├── docker-entrypoint.sh
    └── env.dev
```

#### Data

For finetuning the model, we gathered a dataset of 658 mental health-related Q&A conversations, consisting of 594 training samples and 64 test samples. The dataset includes conversations covering various mental health topics, such as anxiety and depression, and is structured to provide emotional support through chatbot responses. The dataset was sourced from mental health FAQ, classical therapy conversations, and general advice interactions. It has been preprocessed and stored in a format suitable for fine-tuning models, which enables a chatbot to assist users with mental health concerns by identifying intents and providing appropriate responses.

For the RAG model, we gathered 7 academic papers on psychological and mental health counseling in .txt format. This dataset enhances the chatbot's ability to provide accurate, research-based mental health support. This dataset serves as a key resource for enhancing the chatbot's ability to provide accurate and contextually relevant mental health support.
   
## Data Pipeline Overview

1. **`src/finetune_data/prepare_data.py`**
   This script handles preprocessing on our mental health dataset with 658 mental health-related Q&A conversations. The preprocessed train and test dataaets arestored on GCS.

2. **`src/rag_data_pipeline/dataloader.py`**
   This script prepares the necessary data for setting up our vector database. It downloads pychological paper raw data from the GCS.

3. **`src/rag_data_pipeline/preprocess_rag.py``**
   This script prepares the necessary data for setting up our vector database. It performs chunking, embedding and loading data to the ChromaDB database in the localhost.

4. **`src/**/Pipfile`**
   We use the following packages to help:
   - user-agent
   - requests
   - google-cloud-storage
   - google-generativeai
   - google-cloud-aiplatform
   - pandas
   - scikit-learn
   - langchain
   - llama-index
   - chromadb
   - langchain-community

5. **`src/**/Dockerfile`**
   Our Dockerfiles follow standard conventions, with the exception of some specific modifications described in the Dockerfile part below.

## Docker and Containerization

**Running Dockerfile**
To run Dockerfile, navigate to the target folder and run `sh docker-shell.sh`.

**Containers**
1. Data Preparation Container: This container processes the dataset, preparing it for model fine-tuning. It works with a labeled dataset of questions and responses.

	**Input:** Raw datasets and environment variables (e.g., GCP project name, GCS bucket name) stored in `env.dev`.

	**Output:** Generates `train.jsonl` and `test.jsonl` files, which are then uploaded to the specified GCS bucket.

2. Model Fine-Tuning Container: This container performs the model fine-tuning process.

	**Input:** The `train.jsonl` and `test.jsonl` stored on GCS.

	**Output:** A fine-tuned model deployed to a Google Cloud endpoint.

3. RAG Workflow Preparation Container: This container handles the setup for the Retrieval-Augmented Generation (RAG) workflow, including downloading raw data from GCS, chunking it, generating embeddings, and populating the vector database.

	**Input:** Source and destination GCS locations, along with any necessary secrets (passed via Docker).

	**Output:** A ChromaDB database stored locally.

4. RAG Workflow Execution Container: This container runs the RAG workflow using the fine-tuned model.

	**Input:** The fine-tuned model endpoint on GCP.

	**Output:** A chatbot that processes user inputs and generates responses using the fine-tuned LLM.


## Data Versioning 


We maintain a history of prompt changes through Git's version control, allowing us to manage updates, compare iterations, and revert to previous versions if necessary. Each version of a prompt is committed with detailed messages, ensuring transparency in modifications and facilitating collaboration across the team. 
----