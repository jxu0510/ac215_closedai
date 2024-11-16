from unittest.mock import patch

@patch('app.generate_query_embedding')
@patch('app.client.get_collection')
@patch('app.generative_model.generate_content')
def test_chat_route(mock_generate_content, mock_get_collection, mock_generate_query_embedding, client):
    """Test the /chat API route."""
    # Mock return values
    mock_generate_query_embedding.return_value = [0.1, 0.2, 0.3]
    mock_get_collection.return_value.query.return_value = {'documents': [["doc1", "doc2"]]}
    mock_generate_content.return_value.text = "This is a generated response."

    # Simulate API call
    response = client.post('/chat', json={'message': 'Hello!'})
    data = response.get_json()

    # Assertions
    assert response.status_code == 200
    assert 'reply' in data
    assert data['reply'] == "This is a generated response."
