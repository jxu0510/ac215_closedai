from unittest.mock import patch
from app import generate_query_embedding, generate_response
from unittest.mock import MagicMock

@patch('app.embedding_model.get_embeddings')
def test_generate_query_embedding(mock_get_embeddings):
    """Test the generate_query_embedding function."""
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3] 
    mock_get_embeddings.return_value = [mock_embedding]
    result = generate_query_embedding("test query")
    assert result == [0.1, 0.2, 0.3]

@patch('app.generative_model.generate_content')
def test_generate_response(mock_generate_content):
    """Test the generate_response function."""
    mock_generate_content.return_value.text = "Generated response"
    generation_config = {"max_output_tokens": 8192, "temperature": 0.25, "top_p": 0.95}
    result = generate_response("Test input", generation_config)
    assert result.text == "Generated response"
