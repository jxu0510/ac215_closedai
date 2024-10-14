import os
import argparse
import pandas as pd
import json
import glob
from sklearn.model_selection import train_test_split
from google.cloud import storage

# Setup
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = "us-central1"
OUTPUT_FOLDER = "../../data"
GCS_BUCKET_NAME = os.environ["GCS_BUCKET_NAME"]

# Function to prepare the dataset
def prepare():
    print("Preparing dataset...")

    # Load your dataset (Adjust the path to your dataset)
    with open(os.path.join(OUTPUT_FOLDER, "data.json"), "r") as file:
        data = json.load(file)

    # List to store all the pattern-response pairs
    pairs = []

    # Iterate through intents in the dataset
    for intent in data['intents']:
        patterns = intent['patterns']
        responses = intent['responses']

        # Match each pattern with each response
        for pattern in patterns:
            for response in responses:
                pairs.append({
                    "question": pattern,
                    "answer": response
                })

    # Convert the pairs into a DataFrame for easier handling
    df = pd.DataFrame(pairs)

    # Perform train/test split (90% training, 10% testing)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Prepare the fine-tuning format
    def convert_to_finetune_format(df):
        formatted = []
        for _, row in df.iterrows():
            entry = {
                "contents": [
                    {"role": "user", "parts": [{"text": row['question']}]},
                    {"role": "model", "parts": [{"text": row['answer']}]}
                ]
            }
            formatted.append(entry)
        return formatted

    # Convert training and test data to fine-tuning format
    train_data = convert_to_finetune_format(train_df)
    test_data = convert_to_finetune_format(test_df)

    # Ensure output folder exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Save the data in JSONL format
    with open(os.path.join(OUTPUT_FOLDER, "train.jsonl"), "w") as train_file:
        for item in train_data:
            train_file.write(json.dumps(item) + "\n")

    with open(os.path.join(OUTPUT_FOLDER, "test.jsonl"), "w") as test_file:
        for item in test_data:
            test_file.write(json.dumps(item) + "\n")

    print("Data preparation complete!")

# Function to upload the prepared files to Google Cloud Storage
def upload():
    print("Uploading data...")

    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    timeout = 300

    data_files = glob.glob(os.path.join(OUTPUT_FOLDER, "*.jsonl")) + glob.glob(os.path.join(OUTPUT_FOLDER, "*.csv"))
    data_files.sort()

    # Upload files to GCS
    for index, data_file in enumerate(data_files):
        filename = os.path.basename(data_file)
        destination_blob_name = os.path.join("llm-finetune-dataset", filename)
        blob = bucket.blob(destination_blob_name)
        print(f"Uploading {data_file} to {destination_blob_name}...")
        blob.upload_from_filename(data_file, timeout=timeout)

    print("Upload complete!")

# Main function to handle arguments
def main(args=None):
    print("CLI Arguments:", args)

    if args.prepare:
        prepare()

    if args.upload:
        upload()

# Parse command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for LLM Fine-tuning")

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare the dataset for fine-tuning",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload prepared data to GCS",
    )

    args = parser.parse_args()

    main(args)
