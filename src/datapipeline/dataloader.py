import os
import shutil
from google.cloud import storage

gcp_project = "ac215-438023"
bucket_name = "psychology_ref"


def makedirs():
    os.makedirs("../../data", exist_ok = True)

def download():
    print("downloading")

    shutil.rmtree("../../data", ignore_errors=True, onerror=None)
    makedirs()

    client = storage.Client(project=gcp_project)
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs()
    for blob in blobs:
        print(blob.name)
        if not blob.name.endswith("/"):
            blob.download_to_filename("../../data/" + blob.name)

def main():
    download()

if __name__ == "__main__":
    main()