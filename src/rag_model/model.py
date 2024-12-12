import os
import argparse
import chromadb


import vertexai
from vertexai.language_models import TextEmbeddingInput
from vertexai.generative_models import GenerativeModel
from vertexai.preview.language_models import TextEmbeddingModel


# Setup
# GENERATIVE_MODEL = "gemini-1.5-flash-001"
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSION = 256
INPUT_FOLDER = "/app/data"
OUTPUT_FOLDER = "/app/outputs"
CHROMADB_HOST = "llm-rag-chromadb"
CHROMADB_PORT = 8000
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)
# https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#python
embedding_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
# Configuration settings for the content generation
generation_config = {
    "max_output_tokens": 8192,  # Maximum number of tokens for output
    "temperature": 0.25,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}
# Initialize the GenerativeModel with specific system instructions
SYSTEM_INSTRUCTION = """
You are an AI assistant specialized in psychology, focused on providing empathetic, supportive, and informative responses.

Your goal is to help the user by drawing on the information provided in the referenced materials.
You may use only that information to inform your understanding,
but do not mention these sources or refer to them directly.
Instead, speak directly and compassionately to the user as if you're having a one-on-one conversation.
Maintain an empathetic, understanding tone, and give clear, practical suggestions when possible.

When responding:
1. Directly address the user's feelings and concerns in a caring, supportive manner.
2. Use the information you have to offer guidance, suggestions, or insights that could help them.
3. If there's insufficient information to fully answer their question,
gently express that you understand how complex their situation might be and encourage them to seek additional support.
4. Stay focused on providing psychologically sound advice without being overly formal or academic.
Aim to be warm, understanding, and constructive.
"""

# Baseline model
# generative_model = GenerativeModel(
# 	GENERATIVE_MODEL,
# 	system_instruction=[SYSTEM_INSTRUCTION]
# )

MODEL_ENDPOINT = (
    "projects/xenon-depth-434717-n0/locations/us-central1/endpoints/6946258166963240960"
)

generative_model = GenerativeModel(
    MODEL_ENDPOINT, system_instruction=None
)


def generate_query_embedding(query):
    query_embedding_inputs = [
        TextEmbeddingInput(task_type="RETRIEVAL_DOCUMENT", text=query)
    ]
    kwargs = (
        dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
    )
    embeddings = embedding_model.get_embeddings(query_embedding_inputs, **kwargs)
    return embeddings[0].values


def chat(method="char-split"):
    print("chat()")

    # Connect to chroma DB
    client = chromadb.HttpClient(host=CHROMADB_HOST, port=CHROMADB_PORT)
    # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
    collection_name = f"{method}-collection"

    while True:

        query = input("Enter your query: ")
        if query.lower() == "exit":
            break
        query_embedding = generate_query_embedding(query)

        # Get the collection
        collection = client.get_collection(name=collection_name)

        # Query based on embedding value
        results = collection.query(query_embeddings=[query_embedding], n_results=10)
        # 	print("\n\nResults:", results)

        # 	print(len(results["documents"][0]))
        #
        joined_results = "\n".join(results["documents"][0])
        INPUT_PROMPT = f"""
            {query}
            {joined_results}
            """

        # print("INPUT_PROMPT: ",INPUT_PROMPT)
        response = generative_model.generate_content(
            [INPUT_PROMPT],  # Input prompt
            generation_config=generation_config,  # Configuration settings
            stream=False,  # Enable streaming for responses
        )
        generated_text = response.text
        print("LLM Response:", generated_text)


def main(args=None):
    print("CLI Arguments:", args)

    if args.chat:
        chat(method=args.chunk_type)


if __name__ == "__main__":
    # Generate the inputs arguments parser
    # if you type into the terminal '--help', it will provide the description
    parser = argparse.ArgumentParser(description="CLI")
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with LLM",
    )
    parser.add_argument(
        "--chunk_type",
        default="char-split",
        help="char-split | recursive-split | semantic-split",
    )

    args = parser.parse_args()

    main(args)
