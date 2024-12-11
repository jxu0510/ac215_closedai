import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_chromadb_client():
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.create_collection.return_value = mock_collection
    mock_client.get_collection.return_value = mock_collection
    return mock_client, mock_collection


@patch("chromadb.HttpClient")
def test_integration(mock_http_client, mock_chromadb_client):
    mock_client, mock_collection = mock_chromadb_client
    mock_http_client.return_value = mock_client
    input_books = ["book_1", "book_2", "book_3"]

    chunks = [
        {"chunk": f"Chunk {i} from {book}", "book": book}
        for i, book in enumerate(input_books, 1)
    ]
    data_df = pd.DataFrame(chunks)

    mock_embeddings = [[0.1 * i] * 256 for i in range(len(chunks))]
    data_df["embedding"] = mock_embeddings

    load_text_embeddings(data_df, mock_collection)
    assert mock_collection.add.called, "Collection add method was not called."
    assert len(mock_collection.add.call_args[1]["ids"]) == len(chunks)

    mock_collection.query.return_value = {
        "results": [{"id": f"result_{i}", "score": 0.9} for i in range(3)]
    }
    results = mock_collection.query(query_embeddings=[[0.1] * 256], n_results=3)

    assert results["results"], "No results returned from the query."
    mock_collection.get.return_value = {
        "documents": [{"id": f"doc_{i}", "content": f"Document {i}"} for i in range(3)]
    }
    retrieved = mock_collection.get(where={"book": "book_1"}, limit=1)

    assert retrieved["documents"], "No documents retrieved."


def load_text_embeddings(df, collection, batch_size=500):
    df["id"] = [f"mock_id_{i}" for i in range(len(df))]
    total_inserted = 0
    for i in range(0, df.shape[0], batch_size):
        batch = df.iloc[i: i + batch_size]
        collection.add(
            ids=batch["id"].tolist(),
            documents=batch["chunk"].tolist(),
            metadatas=[{"book": book} for book in batch["book"].tolist()],
            embeddings=batch["embedding"].tolist(),
        )
        total_inserted += len(batch)
    print(f"Inserted {total_inserted} items into collection.")
