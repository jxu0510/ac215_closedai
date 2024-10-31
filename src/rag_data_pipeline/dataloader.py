import os
import shutil
from google.cloud import storage

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCS_RAG_BUCKET_NAME = "closed-ai-rag"


def makedirs():
    os.makedirs("../../data", exist_ok = True)

def download():
    print("downloading")

    shutil.rmtree("../../data", ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client(project=GCP_PROJECT)
    bucket = client.get_bucket(GCS_RAG_BUCKET_NAME)

    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename("../../data/" + blob.name)

def main():
    download()

if __name__ == "__main__":
    main()