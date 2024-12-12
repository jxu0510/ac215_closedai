# AC215 Final Project - README

![image](https://github.com/user-attachments/assets/eb0e8154-cc16-4b0a-ba80-af884ffa3cc5)

#### Group Name
ClosedAI

#### Team Members
Nina Mao, Yunxiao Tang, Jiayi Xu, Xinjie Yi

#### Project
In this project, we aim to develop an AI-powered mental healing application. The app will feature advanced conversational technology to engage in meaningful dialogue with individuals experiencing negative psychological states. Powered by fine-tuned GEMINI and RAG models, the application is designed to offer specialized mental healing support. Users can interact with the app through natural conversations, where it draws from a wealth of expert psychology literature to provide professional, evidence-based mental health guidance. Whether users are dealing with stress, anxiety, or other emotional challenges, the app offers personalized therapeutic advice, helping them navigate difficult emotions and promote mental well-being.

## Milestone 5

In Milestone 5, we deployed the application to a Kubernetes cluster, demonstrating scalability by manually adjusting load levels. We wrote Ansible playbooks to automate the provisioning and deployment of both the Kubernetes cluster and the application. A CI/CD pipeline was implemented using GitHub Actions to automate testing, integration, and deployment, ensuring 78% test coverage. Finally, we integrated a production-ready ML workflow with automated data preprocessing and model deploying.

Due to the integration with Google Cloud Platform services, certain parts of our codebase cannot be fully tested in a traditional development environment. Specifically, the Retrieval-Augmented Generation (RAG) pipeline, which relies on direct interactions with Google Cloud resources, falls outside the scope of our standard testing procedures. This limitation arises because the RAG pipeline requires live cloud resources, which are not replicable in local or simulated testing environments. As a result, any components that depend on Google Cloud services must be tested in a live cloud environment to ensure functionality and performance accuracy.

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
    ├── deployment
    ├── service
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── app.py
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   ├── static
    │   │   ├── styles.css
    │   │   └── assets
    │   │       └── chat_background.png
    │   └── templates
    │       └── index.html
    ├── vectordb
    │   ├── Dockerfile
    │   ├── Pipfile
    │   ├── Pipfile.lock
    │   ├── cli.py
    │   ├── docker-entrypoint.sh
    │   ├── docker-shell.sh
    │   └── docker-compose.yml
    └── tests
        ├── conftest.py
        ├── test_integrate.py
        ├── test_system.py
        └── test_unit.py
```

## **1. Prerequisites and Setup Instructions**

### **1.1 Prerequisites**
#### Software Requirements
- Docker
- Kubernetes
- Ansible
- Python
- Git
- pipenv

#### Cloud Provider
- A Google Cloud Platform (GCP) project with Compute Engine and Kubernetes Engine enabled.

#### Service Accounts
1. Create a GCP service account with permissions for Kubernetes and Compute Engine.
2. Download the service account key (`.json`).
3. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

### **1.2 Setup Instructions**
#### Clone the Repository
   ```bash
   git clone git@github.com:jxu0510/ac215_closedai.git
   ```
#### Configure Ansible Inventory
Edit the inventory.yml file to define your target hosts and GCP configurations.

#### API's to Enable in GCP
Search for each of these in the GCP search bar and click enable to enable these API's
* Compute Engine API
* Service Usage API
* Cloud Resource Manager API
* Google Container Registry API

#### Setup GCP Service Account
- Go to [GCP Console](https://console.cloud.google.com/home/dashboard), search for  "Service accounts" from the top search box. or go to: "IAM & Admins" > "Service accounts" from the top-left menu and create a new service account called "deployment".
- For `deployment`:
    - Compute Admin
    - Compute OS Login
    - Container Registry Service Agent
    - Kubernetes Engine Admin
    - Service Account User
    - Storage Admin
- Rename the json key file to `deployment.json`
- Follow the same process Create another service account called `gcp-service`
- For `gcp-service` give the following roles:
    - Storage Object Viewer
    - Vertex AI Administrator
- Rename the json key file to `gcp-service.json`

#### Configuring OS Login for Service Account
```
gcloud compute project-info add-metadata --project <YOUR GCP_PROJECT> --metadata enable-oslogin=TRUE
```

#### Create SSH Key for Service Account
```
cd /secrets
ssh-keygen -f ssh-key-deployment
cd /app
```

#### Providing Public SSH Keys to Instances
```
gcloud compute os-login ssh-keys add --key-file=/secrets/ssh-key-deployment.pub
```

## **2. Deployment Instructions**

#### Build and Push Docker Containers to GCR (Google Container Registry)
```
ansible-playbook deploy-docker-images.yml -i inventory.yml
```

#### Create Compute Instance (VM) Server in GCP
```
ansible-playbook deploy-create-instance.yml -i inventory.yml --extra-vars cluster_state=present
```

#### Provision Compute Instance in GCP
```
ansible-playbook deploy-provision-instance.yml -i inventory.yml
```

#### Setup Docker Containers in the Compute Instance
```
ansible-playbook deploy-setup-containers.yml -i inventory.yml
```

#### Configure Nginx File for Web Server
* Create nginx.conf file for defaults routes in web server

#### Setup Webserver on the Compute Instance
```
ansible-playbook deploy-setup-webserver.yml -i inventory.yml
```

## **3. Usage Details**
Once the deployment is complete, the web server will be running on the external IP of the compute instance.

Examples:

Q1: What are some factors affecting adolescent mental health problems?

A1: Based on the provided text, several factors affect adolescent mental health problems: * **Social stress:** This is identified as one of three factors impacting adolescent mental health problems. * **Substance abuse:** This is another of the three factors impacting adolescent mental health problems. * **Student-teacher relationship:** Positive relationships at school are associated with positive mental health outcomes in adolescents. * **Sense of community at school:** This contextual factor is linked to positive mental health outcomes. * **Bullying:** A significant school-specific factor, with bullied adolescents showing more than doubled odds of suicidal ideation and attempts. The text mentions that 46% of school-aged children in a UK survey reported being bullied. The text also mentions that mental health risk factors increase the chances of developing psychological illness, while protective factors can reduce the negative impact of these risks. The text does not elaborate on specific protective factors beyond positive student-teacher relationships and a sense of community at school.

Q2: Can you tell me what are some ways to mitigate the risk of having mental health problems?

A2: Based on the provided text, mitigating the risk of mental health problems involves focusing on both risk and protective factors. The text highlights the importance of preventive interventions where different agents work together to promote children's and adolescents' mental health. These interventions should be culturally appropriate. Furthermore, reducing the negative impact of risk factors is crucial, as these factors can increase the chances of developing psychological illness. The text also mentions the need for a shift away from an individual-focused approach to service delivery, suggesting that a broader, population-level approach may be more effective in decreasing rates of mental illness.

## **4. Known Issues and Limitations**

1. Overdependence on RAG Dataset

The chatbot currently relies heavily on the Retrieval-Augmented Generation (RAG) workflow, resulting in responses that are overly factual and rigid, which lacks the conversational and empathetic tone users expect. This overdependence on academic papers can make response

2. Quality of RAG Dataset

The current RAG workflow is limited by the insufficient quantity and diversity of academic papers on mental health counseling for younger adults. This restricts the chatbot’s ability to address a wide range of queries, even within this demographic. Expanding the dataset to cover more topics will improve the chatbot’s relevance and responsiveness while balancing factual content with conversational flexibilit

3. Limited Model Exploration and Flexibility

The chatbot’s current performance is restricted by our reliance on a single language model—GEMINI. While GEMINI’s integration with GCP made it a good basic model, we have not yet investigated how other large language models, such as GPT and Llama, might influence the quality, tone, or adaptability of responses. Exploring alternative models could provide opportunities for more helpful and domain-specific dialogues.
