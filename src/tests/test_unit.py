import unittest
import os
import dataloader
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open


class TestFineTuningScript(unittest.TestCase):
    @patch("time.sleep", return_value=None)
    @patch("vertexai.preview.tuning.sft.train")
    def test_train(self, mock_sft_train, mock_sleep):
        mock_sft_job = MagicMock()
        mock_sft_job.has_ended = True
        mock_sft_job.tuned_model_name = "mock-tuned-model-name"
        mock_sft_job.tuned_model_endpoint_name = "mock-endpoint-name"
        mock_sft_job.experiment = "mock-experiment"
        mock_sft_train.return_value = mock_sft_job

        from finetune import train
        train(wait_for_job=True)

        mock_sft_train.assert_called_once_with(
            source_model="llama3-70b-hf",
            train_dataset="gs://closed-ai/llm-finetune-dataset/train.jsonl",
            validation_dataset="gs://closed-ai/llm-finetune-dataset/test.jsonl",
            epochs=3,
            adapter_size=4,
            learning_rate_multiplier=1.0,
            tuned_model_display_name="mental-health-chatbot-v1",
        )
        mock_sleep.assert_called()
        self.assertEqual(mock_sft_job.tuned_model_name, "mock-tuned-model-name")


class TestLLMFineTuningData(unittest.TestCase):
    @patch("prepare_data.os.makedirs")
    @patch("prepare_data.train_test_split")
    @patch("prepare_data.open", new_callable=mock_open,
           read_data='{"intents": [{"patterns": ["Hello"], "responses": ["Hi there!"]}]}')
    def test_prepare(self, mock_open_file, mock_train_test_split, mock_makedirs):
        mock_train_test_split.return_value = (pd.DataFrame({"question": ["Hello"], "answer": ["Hi there!"]}),
                                              pd.DataFrame({"question": ["Hi"], "answer": ["Hello there!"]}))
        from prepare_data import prepare
        prepare()

        self.assertTrue(mock_train_test_split.called)

        mock_open_file.assert_any_call("../../data/train.jsonl", "w")
        mock_open_file.assert_any_call("../../data/test.jsonl", "w")


class TestPreprocessRag(unittest.TestCase):
    @patch("preprocess_rag.get_embeddings")
    def test_generate_query_embedding(self, mock_get_embeddings):
        mock_get_embeddings.return_value = [MagicMock(values=[0.1, 0.2, 0.3])]

        from preprocess_rag import generate_query_embedding
        result = generate_query_embedding("test query")
        mock_get_embeddings.assert_called_once()
        self.assertEqual(result, [0.1, 0.2, 0.3])

    @patch("preprocess_rag.get_embeddings")
    def test_generate_text_embeddings(self, mock_get_embeddings):
        mock_get_embeddings.side_effect = lambda inputs, **kwargs: [
            MagicMock(values=[0.1, 0.2, 0.3]) for _ in inputs
        ]

        from preprocess_rag import generate_text_embeddings
        chunks = ["chunk1", "chunk2"]
        result = generate_text_embeddings(chunks)

        mock_get_embeddings.assert_called_once()
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])

    @patch("preprocess_rag.chromadb.HttpClient")
    def test_load(self, mock_chromadb_client):
        mock_client = MagicMock()
        mock_chromadb_client.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection

        df = pd.DataFrame(
            {"chunk": ["chunk1"], "embedding": [[0.1, 0.2, 0.3]], "book": ["mh1"]}
        )

        from preprocess_rag import load_text_embeddings
        load_text_embeddings(df, mock_collection)

        mock_collection.add.assert_called_once()

    @patch("preprocess_rag.chromadb.HttpClient")
    @patch("preprocess_rag.get_embeddings")
    def test_query(self, mock_get_embeddings, mock_chromadb_client):
        mock_get_embeddings.return_value = [MagicMock(values=[0.1, 0.2, 0.3])]

        mock_client = MagicMock()
        mock_chromadb_client.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_collection.query.return_value = {"results": [{"id": "doc1", "score": 0.95}]}

        from preprocess_rag import query
        query(method="char-split")

        mock_get_embeddings.assert_called_once()
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]], n_results=10
        )

    @patch("preprocess_rag.chromadb.HttpClient")
    def test_get(self, mock_chromadb_client):
        mock_client = MagicMock()
        mock_chromadb_client.return_value = mock_client
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_collection.get.return_value = {"results": [{"id": "doc1"}]}

        from preprocess_rag import get
        get(method="char-split")

        mock_collection.get.assert_called_once_with(where={"book": "mh_2"}, limit=10)


class TestDownloadFunction(unittest.TestCase):
    @patch("dataloader.storage.Client")  # Mock the Google Cloud Storage client
    @patch("dataloader.makedirs")       # Mock the `makedirs` function
    @patch("dataloader.shutil.rmtree")  # Mock the `shutil.rmtree` function
    def test_download(self, mock_rmtree, mock_makedirs, mock_storage_client):
        os.environ["GCP_PROJECT"] = "test-project"

        mock_client_instance = MagicMock()
        mock_storage_client.return_value = mock_client_instance
        mock_bucket = MagicMock()
        mock_client_instance.get_bucket.return_value = mock_bucket

        mock_blob1 = MagicMock()
        mock_blob1.name = "file1.txt"
        mock_blob2 = MagicMock()
        mock_blob2.name = "folder1/"
        mock_blob3 = MagicMock()
        mock_blob3.name = "file2.txt"
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]

        dataloader.download()

        mock_rmtree.assert_called_once_with("../../data", ignore_errors=True, onerror=None)
        mock_makedirs.assert_called_once_with()
        mock_client_instance.get_bucket.assert_called_once_with("closed-ai")
        mock_bucket.list_blobs.assert_called_once()


class AdditionalPreprocessRagTests(unittest.TestCase):
    def setUp(self):
        # Common setup for RAG-related tests
        os.environ["GCP_PROJECT"] = "test-project"
        os.environ["GCP_LOCATION"] = "us-central1"

    def test_count_txt_files(self):
        """Test the count_txt_files function."""
        from preprocess_rag import count_txt_files

        # Mock glob to return specific files
        with patch('glob.glob', return_value=['file1.txt', 'file2.txt', 'file3.txt']):
            file_count = count_txt_files('/mock/folder')
            self.assertEqual(file_count, 3)

    def test_create_book_mappings(self):
        """Test the create_book_mappings function."""
        from preprocess_rag import create_book_mappings

        # Mock count_txt_files to return specific number of files
        with patch('preprocess_rag.count_txt_files', return_value=3):
            mappings = create_book_mappings('/mock/folder')

            self.assertEqual(len(mappings), 3)
            self.assertEqual(mappings['mh1']['author'], 'the 1st book')
            self.assertEqual(mappings['mh2']['author'], 'the 2nd book')
            self.assertEqual(mappings['mh3']['author'], 'the 3rd book')
            self.assertEqual(mappings['mh1']['year'], 2023)

    def test_embed_method(self):
        """Test the embed function."""
        from preprocess_rag import embed

        # Mock dependencies
        with patch('glob.glob', return_value=['/mock/chunks-char-split-file1.jsonl']), \
             patch('pandas.read_json', return_value=pd.DataFrame({'chunk': ['test chunk']})), \
             patch('preprocess_rag.generate_text_embeddings', return_value=[[0.1, 0.2]]), \
             patch('builtins.open', mock_open()) as mock_file:

            embed(method='char-split')

            # Verify embeddings were generated
            mock_file.assert_called()

    @patch('preprocess_rag.chromadb.HttpClient')
    def test_load_with_error_handling(self, mock_chromadb_client):
        """Test load function with collection deletion error handling."""
        from preprocess_rag import load

        # Setup mocks
        mock_client = MagicMock()
        mock_chromadb_client.return_value = mock_client

        # Simulate an error when deleting collection
        mock_client.delete_collection.side_effect = Exception("Collection deletion failed")

        # Mock glob and other dependencies
        with patch('glob.glob', return_value=['/mock/embeddings-char-split-file1.jsonl']), \
             patch('pandas.read_json', return_value=pd.DataFrame({
                 'chunk': ['test chunk'],
                 'embedding': [[0.1, 0.2]],
                 'book': ['mh1']
             })):

            # This should not raise an exception
            load(method='char-split')

    def test_ordinal_function(self):
        """Test the ordinal function for various inputs."""
        from preprocess_rag import ordinal

        test_cases = [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (4, "4th"),
            (11, "11th"),
            (21, "21st"),
            (22, "22nd"),
            (23, "23rd"),
            (100, "100th")
        ]

        for number, expected in test_cases:
            self.assertEqual(ordinal(number), expected)

    def test_get_method(self):
        """Test the get method with additional assertions."""
        from preprocess_rag import get

        # Mock ChromaDB client and collection
        with patch('preprocess_rag.chromadb.HttpClient') as mock_chromadb_client:
            mock_client = MagicMock()
            mock_chromadb_client.return_value = mock_client
            mock_collection = MagicMock()
            mock_client.get_collection.return_value = mock_collection

            # Simulate get method call
            get(method='char-split')

            # Verify correct method calls
            mock_client.get_collection.assert_called_once_with(name='char-split-collection')
            mock_collection.get.assert_called_once_with(where={"book": "mh_2"}, limit=10)


class AdditionalErrorHandlingTests(unittest.TestCase):
    def test_generate_query_embedding_with_empty_input(self):
        """Test query embedding generation with empty input."""
        from preprocess_rag import generate_query_embedding

        with patch('preprocess_rag.get_embeddings', return_value=[MagicMock(values=[])]):
            result = generate_query_embedding("")
            self.assertEqual(result, [])

    def test_generate_text_embeddings_with_empty_list(self):
        """Test text embeddings generation with empty input."""
        from preprocess_rag import generate_text_embeddings

        with patch('preprocess_rag.get_embeddings', return_value=[]):
            result = generate_text_embeddings([])
            self.assertEqual(result, [])
