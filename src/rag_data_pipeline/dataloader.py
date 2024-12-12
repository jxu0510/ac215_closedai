import os
import shutil
from google.cloud import storage
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_RAG_BUCKET_NAME = "closed-ai"
RAG_FOLDER = "llm-rag-dataset"


def makedirs():
    os.makedirs("../../data", exist_ok=True)


def download():
    print("downloading")

    shutil.rmtree("../../data", ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.get_bucket(GCS_RAG_BUCKET_NAME)

    blobs = bucket.list_blobs()
    for blob in blobs:
        # Only download files inside the specified folder
        if blob.name.startswith(RAG_FOLDER) and not blob.name.endswith("/"):
            relative_path = blob.name[len(RAG_FOLDER) + 1:]
            local_path = os.path.join("./data", relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"Downloading {blob.name} to {local_path}")
            blob.download_to_filename(local_path)


def main():
    download()


if __name__ == "__main__":
    main()
