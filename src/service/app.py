import os
import argparse
from flask import Flask, request, jsonify, render_template
# Setup Flask App
app = Flask(__name__)
# Setup Vertex AI configurations
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000
MODEL_ENDPOINT = (
        "projects/xenon-depth-434717-n0/locations/us-central1/endpoints/6946258166963240960"
)

    # Initialize the generative model with system instructions
SYSTEM_INSTRUCTION = (
        """ You are an AI assistant specialized in psychology. Your responses are based """
        """solely on the information provided in the text chunks given to you. Do not use """
        """any external knowledge..."""
)
# Embedding generation function
def generate_query_embedding(query):
    embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    query_embedding_inputs = [
        TextEmbeddingInput(task_type="RETRIEVAL_DOCUMENT", text=query)
    ]
    kwargs = (
        dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
    )
    embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
    rst = embeddings[0].values
    return rst

def generate_response(input_prompt, generation_config):
    generative_model = GenerativeModel(
        MODEL_ENDPOINT, system_instruction=[SYSTEM_INSTRUCTION]
    )
    response = generative_model.generate_content([input_prompt], generation_config=generation_config, stream=False)
    text = response.text
    return text

def get_doc_from_client(collection_name, query_embedding):
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    collection = client.get_collection(name=collection_name)
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    return results

@app.route('/')
def index():
    return render_template('index.html')


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    query = data.get("message", "")
    if not query:
        return jsonify({"reply": "Please enter a message."})

    try:
        query_embedding = generate_query_embedding(query)

        # Retrieve collection and query the DB
        collection_name = "char-split-collection"
        results = get_doc_from_client(collection_name, query_embedding)

        # Construct input prompt from query and results
        documents = "\n".join(results["documents"][0])
        input_prompt = f"{query}\n{documents}"

        # Generate response using the model
        generation_config = {"max_output_tokens": 8192, "temperature": 0.25, "top_p": 0.95}
        response = generate_response(input_prompt, generation_config)

        generated_text = response
        print(generated_text)
        return jsonify({"reply": generated_text})

    except Exception as e:
        print(e)
        return jsonify({"reply": "Error: Could not generate a response."}), 500


if __name__ == "__main__":
    import chromadb
    import vertexai
    from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
    from vertexai.generative_models import GenerativeModel

    vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
    app.run(host="0.0.0.0", port=8080, debug=True)
