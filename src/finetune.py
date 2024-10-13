import os
import argparse
import time
from google.cloud import storage
import vertexai
from vertexai.preview.tuning import sft
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Setup your GCP Project and dataset paths
GCP_PROJECT = os.environ["GCP_PROJECT"]
TRAIN_DATASET = "gs://closed-ai/llm-finetune-dataset/train.jsonl"
VALIDATION_DATASET = "gs://closed-ai/llm-finetune-dataset/test.jsonl"
GCP_LOCATION = "us-central1"
GENERATIVE_SOURCE_MODEL = "gemini-1.5-flash-002"  # Replace with model like llama-2-7b or flan-t5

# Configuration settings for fine-tuning
generation_config = {
    "max_output_tokens": 3000,  # Maximum number of tokens for input
    "temperature": 0.75,  # Control randomness in output
    "top_p": 0.95,  # Use nucleus sampling
}

# Initialize Vertex AI environment
vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION)

def train(wait_for_job=False):
    print("Starting fine-tuning...")

    # Start fine-tuning the model
    sft_tuning_job = sft.train(
        source_model=GENERATIVE_SOURCE_MODEL,
        train_dataset=TRAIN_DATASET,
        validation_dataset=VALIDATION_DATASET,
        epochs=3,  # Change number of epochs based on dataset size
        adapter_size=4,  # Adapter size to reduce number of trainable parameters
        learning_rate_multiplier=1.0,  # Modify this for learning rate control
        tuned_model_display_name="mental-health-chatbot-v1",  # Change this to your tuned model name
    )

    print("Fine-tuning job started. Monitoring progress...")

    # Wait and refresh job status
    time.sleep(60)
    sft_tuning_job.refresh()

    if wait_for_job:
        print("Check status of tuning job:")
        print(sft_tuning_job)
        while not sft_tuning_job.has_ended:
            time.sleep(60)
            sft_tuning_job.refresh()
            print("Job in progress...")

    print(f"Tuned model name: {sft_tuning_job.tuned_model_name}")
    print(f"Tuned model endpoint name: {sft_tuning_job.tuned_model_endpoint_name}")
    print(f"Experiment: {sft_tuning_job.experiment}")


def chat():
    print("Starting chat with fine-tuned model...")

    # Use the fine-tuned model endpoint (replace with the actual endpoint from your GCP console)
    MODEL_ENDPOINT = "projects/xenon-depth-434717-n0/locations/us-central1/endpoints/6946258166963240960"
    
    # Initialize the fine-tuned model
    generative_model = GenerativeModel(MODEL_ENDPOINT)

    # Test the model with a mental health-related query
    query = "I feel stressed and anxious, what should I do?"
    print("User query: ", query)

    # Generate the response using the fine-tuned model
    response = generative_model.generate_content(
        [query],  # User's query
        generation_config=generation_config,  # Generation configuration
        stream=False,  # Enable streaming if needed
    )
    generated_text = response.text
    print("Fine-tuned LLM Response:", generated_text)


def main(args=None):
    print("Command line arguments:", args)

    if args.train:
        train()

    if args.chat:
        chat()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Fine-tuning CLI for Mental Health Chatbot")

    # Argument to trigger the fine-tuning process
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model using fine-tuning on Vertex AI",
    )

    # Argument to test chat functionality with the fine-tuned model
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Chat with the fine-tuned model",
    )

    # Parse arguments from the command line
    args = parser.parse_args()

    # Run the main function
    main(args)
