import os
import argparse
from flask import Flask, request, jsonify, render_template
import chromadb
import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Setup Flask App
app = Flask(__name__)

# Setup Vertex AI configurations
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000

vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

MODEL_ENDPOINT = "projects/xenon-depth-434717-n0/locations/us-central1/endpoints/6946258166963240960"

# Initialize the generative model with system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in psychology. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge...
"""
generative_model = GenerativeModel(MODEL_ENDPOINT, system_instruction=[SYSTEM_INSTRUCTION])

# Connect to Chroma DB
client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)

# Embedding generation function
def generate_query_embedding(query):
    query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
    kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
    embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
    return embeddings[0].values

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('message', '')

    if not query:
        return jsonify({'reply': 'Please enter a message.'})

    try:
        query_embedding = generate_query_embedding(query)

        # Retrieve collection and query the DB
        collection_name = "char-split-collection"
        collection = client.get_collection(name=collection_name)
        results = collection.query(query_embeddings=[query_embedding], n_results=10)

        # Construct input prompt from query and results
        documents = "\n".join(results['documents'][0])
        input_prompt = f"{query}\n{documents}"

        # Generate response using the model
        generation_config = {"max_output_tokens": 8192, "temperature": 0.25, "top_p": 0.95}
        response = generative_model.generate_content([input_prompt], generation_config=generation_config, stream=False)

        generated_text = response.text
        return jsonify({'reply': generated_text})

    except Exception as e:
        print(e)
        return jsonify({'reply': 'Error: Could not generate a response.'}), 500

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chatbot Interface")
    parser.add_argument("--chat", action="store_true", help="Chat with LLM")
    parser.add_argument("--chunk_type", default="char-split", help="Split type: char-split | recursive-split | semantic-split")
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=5000, debug=True)
