# AC215 Final Project - README

#### Group Name
ClosedAI

#### Team Members
Nina Mao, Yunxiao Tang, Jiayi Xu, Xinjie Yi

#### Project
In this project, we aim to develop an AI-powered mental healing application. The app will feature advanced conversational technology to engage in meaningful dialogue with individuals experiencing negative psychological states. Powered by fine-tuned GEMINI and RAG models, the application is designed to offer specialized mental healing support. Users can interact with the app through natural conversations, where it draws from a wealth of expert psychology literature to provide professional, evidence-based mental health guidance. Whether users are dealing with stress, anxiety, or other emotional challenges, the app offers personalized therapeutic advice, helping them navigate difficult emotions and promote mental well-being.

## Milestone 5

In Milestone 4, we successfully implemented Continuous Integration (CI) using GitHub Actions, automating the build, linting, and testing processes to ensure seamless validation of new code merges. We also implemented tests and reached a coverage rate of 59.2%.

#### Project Organization

```
├── LICENSE
├── Pipfile
├── README.md
├── reports
│   ├── ac215_ms2_deliverable.pdf
│   ├── ac215_ms3_deliverable.pdf
│   └── mockup.pdf
└── src
    ├── data_versioning
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── rag_dataset
    │   └── rag_dataset.dvc
    ├── docker-compose.yml
    ├── docker-entrypoint.sh
    ├── env.dev
    ├── environment.yaml
    ├── finetune_data
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   └── prepare_data.py
    ├── finetune_model
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   └── finetune.py
    ├── rag_data_pipeline
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── dataloader.py
    │   ├── docker-shell.sh
    │   └── preprocess_rag.py
    ├── rag_model
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── docker-shell.sh
    │   └── model.py
    ├── service
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── app.py
    │   ├── docker-shell.sh
    │   └── templates
    │       └── index.html
    └── tests
        ├── conftest.py
        ├── test_integrate.py
        ├── test_system.py
        └── test_unit.py
```

#### Data

For finetuning the model, we gathered a dataset of 658 mental health-related Q&A conversations, consisting of 594 training samples and 64 test samples. The dataset includes conversations covering various mental health topics, such as anxiety and depression, and is structured to provide emotional support through chatbot responses. The dataset was sourced from mental health FAQ, classical therapy conversations, and general advice interactions. It has been preprocessed and stored in a format suitable for fine-tuning models, which enables a chatbot to assist users with mental health concerns by identifying intents and providing appropriate responses.

For the RAG model, we curated a dataset of academic papers on psychological and mental health counseling specifically targeting teenagers and younger adults. These papers, stored in .txt format, were carefully selected to focus on mental health challenges and therapeutic approaches relevant to this demographic. This dataset enhances the chatbot's ability to provide accurate, research-based mental health support tailored to the needs of teenagers and younger adults, ensuring the responses are contextually relevant and specialized for their unique emotional and psychological challenges.


## Data Pipeline Overview

1. **`src/finetune_data/prepare_data.py`**
   This script handles preprocessing on our mental health dataset with 658 mental health-related Q&A conversations. The preprocessed train and test dataaets arestored on GCS.

2. **`src/rag_data_pipeline/dataloader.py`**
   This script prepares the necessary data for setting up our vector database. It downloads pychological paper raw data from the GCS.

3. **`src/rag_data_pipeline/preprocess_rag.py`**
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
- navigate to the target folder.
- run `pipenv install` to create Pipfile.lock.
- run `sh docker-shell.sh`.

**Containers**

1. **Data Preparation Container**
   This container preprocesses raw datasets to create `train.jsonl` and `test.jsonl` files, which are then uploaded to a Google Cloud Storage (GCS) bucket for model fine-tuning.
   - **Input:** Raw datasets and environment variables (e.g., GCP project name, GCS bucket name) stored in `env.dev`.
   - **Output:** Preprocessed datasets (`train.jsonl`, `test.jsonl`) stored in GCS.

2. **Model Fine-Tuning Container**
   This container handles the fine-tuning process for the model using the preprocessed data.
   - **Input:** Preprocessed datasets (`train.jsonl` and `test.jsonl`) stored in GCS.
   - **Output:** A fine-tuned model deployed to a Google Cloud endpoint.

3. **RAG Workflow Preparation Container**
   This container prepares data for the Retrieval-Augmented Generation (RAG) workflow by downloading raw academic papers from GCS, chunking the data, generating embeddings, and populating the ChromaDB vector database.
   - **Input:** Source and destination GCS locations and secrets passed via `env.dev`.
   - **Output:** A ChromaDB database stored locally.

4. **RAG Workflow Execution Container**
   This container executes the RAG workflow using the fine-tuned model and vector database to generate chatbot responses.
   - **Input:** The fine-tuned model endpoint deployed on GCP and the ChromaDB vector database.
   - **Output:** Contextually relevant chatbot responses.

5. **Data Versioning Container**
   This container manages data versioning for the datasets using DVC (Data Version Control). It facilitates tracking changes to the dataset, storing metadata, and ensuring reproducibility.
   - **Input:** Raw data and versioning configuration files (e.g., `.dvc`, `.dvcignore`).
   - **Output:** Updated versioned datasets and metadata stored in GCS or local storage.

6. **Service Container**
   This container serves the chatbot application as a web-based interface for user interaction. It connects with the fine-tuned model and RAG workflow.
   - **Input:** User queries via the web interface and backend connections to the model and database.
   - **Output:** Chatbot responses displayed on the web application.
   - **Access the Application:**
     Open [http://localhost:8000](http://localhost:8000) in your browser to interact with the chatbot.

**General Note:**
Make sure you have Docker installed and that the required `env.dev` file is present in the respective directories before running any containers.


## Data Versioning

We maintain a history of data version changes using DVC (Data Version Control) specifically for managing the datasets used in the Retrieval-Augmented Generation (RAG) workflow. DVC is utilized to track and manage larger data files, including raw academic papers, preprocessed chunks, and vector embeddings. This setup enables us to track changes in the RAG datasets, compare iterations, and revert to previous versions if necessary.

All updates to the RAG datasets are managed directly through DVC, ensuring transparency and facilitating collaboration across the team.


## Front-end Application

We implement a web-based interface designed to facilitate user interaction with the chatbot. It acts as the primary point of communication where users can submit their queries. The interface processes these inputs, communicates with the backend to connect with the fine-tuned model and RAG workflow, and displays the generated chatbot responses to the user. The application is accessible via a local URL and is optimized for seamless interaction, delivering real-time query-response exchanges.

Users can input mental-health related inquiries to the "type your message" section and click send to receive real-time responses from out chatbot.


## Test Documentation

### Testing Tools

- **PyTest**: Utilized for testing integration and system functionalities.
- **Unittest**: Used for unit testing individual modules and their interactions.
- **Mock**: Used extensively for simulating external dependencies, such as VertexAI and Chromadb to isolate test environments.

### Testing Strategy

#### 1. Unit Tests

Unit tests validate individual components in isolation:

##### `TestFineTuningScript`

- **`test_train`**: Ensures the fine-tuning process is executed with correct parameters and produces expected outputs.

##### `TestLLMFineTuningData`

- **`test_prepare`**: Verifies finetunedata preparation splits intents into training and testing datasets and writes them to correct files.

##### `TestPreprocessRag`

- **`test_generate_query_embedding`**: Tests query embedding generation and ensures embeddings match expected structure.
- **`test_generate_text_embeddings`**: Validates text embedding generation for multiple input chunks.
- **`test_load`**: Ensures embeddings, chunks, and metadata are correctly added to the ChromaDB collection.
- **`test_query`**: Tests querying embeddings and validates results returned from the ChromaDB collection.
- **`test_get`**: Verifies document retrieval from ChromaDB using filters.

##### `TestDownloadFunction`

- **`test_download`**: Ensures data for the RAG system is correctly downloaded from Google Cloud Storage and handled appropriately.

#### 2. Integration Tests

Integration tests ensure that multiple components work together as expected:

- **`test_integration`**: Validates interactions with the ChromaDB client in the RAG process by testing the following:
  - Text embeddings are correctly loaded into the collection with proper metadata and IDs.
  - Queries return expected results from the collection.
  - Document retrieval works with specified filters and returns the expected data.

#### 3. System Tests

System tests Covering user flows and interactions:

- **`test_chat_route`**: Pretends to be a user and tests the `/chat` endpoint for a valid input message and verifies the correct response is generated.

- **`test_no_input_route`**: Pretends to be a user and tests the `/chat` endpoint for a missing input message and ensures an appropriate error response is returned.

### Test Coverage Report

Below is the code coverage summary from the most recent test suite run:

```plaintext
---------- coverage: platform darwin, python 3.12.0-final-0 ----------
Name                                  Stmts   Miss  Cover
---------------------------------------------------------
finetune_data/prepare_data.py            66     23    65%
finetune_model/finetune.py               50     21    58%
rag_data_pipeline/dataloader.py          26      8    69%
rag_data_pipeline/preprocess_rag.py     185    101    45%
service/app.py                           58     24    59%
tests/conftest.py                        12      0   100%
tests/test_integrate.py                  37      0   100%
tests/test_system.py                     25      0   100%
tests/test_unit.py                      102      0   100%
---------------------------------------------------------
TOTAL                                   561    177    68%
```

### Instructions to Run Tests Manually

Follow these steps to replicate the test results locally:

1. **Navigate to the `src` directory**: `cd src`

2. **Export the environment variables**: `source env.dev`

3. **Install the conda environment**: `conda env create -f environment.yaml`

4. **Generate the test report**: `pytest --cov=. tests/`


## Known issues and limitations

1. Overdependence on RAG Dataset
The chatbot currently relies heavily on the Retrieval-Augmented Generation (RAG) workflow, resulting in responses that are overly factual and rigid, which lacks the conversational and empathetic tone users expect. This overdependence on academic papers can make response

2. Quality of RAG Dataset
The current RAG workflow is limited by the insufficient quantity and diversity of academic papers on mental health counseling for younger adults. This restricts the chatbot’s ability to address a wide range of queries, even within this demographic. Expanding the dataset to cover more topics will improve the chatbot’s relevance and responsiveness while balancing factual content with conversational flexibilit

3. Limited Model Exploration and Flexibility
The chatbot’s current performance is restricted by our reliance on a single language model—GEMINI. While GEMINI’s integration with GCP made it a good basic model, we have not yet investigated how other large language models, such as GPT and Llama, might influence the quality, tone, or adaptability of responses. Exploring alternative models could provide opportunities for more helpful and domain-specific dialogues.
