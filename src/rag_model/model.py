import os
import argparse
import pandas as pd
import glob
import hashlib
import chromadb


import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from vertexai.generative_models import GenerativeModel, GenerationConfig, Content, Part, ToolConfig
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
You are an AI assistant specialized in psychology. Your responses are based solely on the information provided in the text chunks given to you. Do not use any external knowledge or make assumptions beyond what is explicitly stated in these chunks.

When answering a query:
1. Carefully read all the text chunks provided.
2. Identify the most relevant information from these chunks to address the user's question.
3. Formulate your response making use of the information found in the given chunks.
4. If the provided chunks do not contain sufficient information to answer the query, state that you don't have enough information to provide a complete answer.
5. Always maintain a professional and knowledgeable tone, befitting a psychology expert.
6. If there are contradictions in the provided chunks, mention this in your response and explain the different viewpoints presented.

Remember:
- Talk to the user and reference to the information in the provided chunks if necessary.
- If asked about topics unrelated to psychology, politely redirect the conversation back to psychology-related subjects.
- Be concise in your responses while ensuring you cover all relevant information from the chunks.
- Solve the problem in the queries, don't mention the provided chunks in the response.

Your goal is to provide accurate, helpful information about psychology based solely on the content of the text chunks you receive with each query.
"""
# generative_model = GenerativeModel(
# 	GENERATIVE_MODEL,
# 	system_instruction=[SYSTEM_INSTRUCTION]
# )

MODEL_ENDPOINT = "projects/xenon-depth-434717-n0/locations/us-central1/endpoints/6946258166963240960"

generative_model = GenerativeModel(MODEL_ENDPOINT, system_instruction=[SYSTEM_INSTRUCTION])

def generate_query_embedding(query):
	query_embedding_inputs = [TextEmbeddingInput(task_type='RETRIEVAL_DOCUMENT', text=query)]
	kwargs = dict(output_dimensionality=EMBEDDING_DIMENSION) if EMBEDDING_DIMENSION else {}
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
		results = collection.query(
			query_embeddings=[query_embedding],
			n_results=10
		)
#	print("\n\nResults:", results)

#	print(len(results["documents"][0]))

		INPUT_PROMPT = f"""
		{query}
		{"\n".join(results["documents"][0])}
		"""

		#print("INPUT_PROMPT: ",INPUT_PROMPT)
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
	parser.add_argument("--chunk_type", default="char-split", help="char-split | recursive-split | semantic-split")

	args = parser.parse_args()

	main(args)