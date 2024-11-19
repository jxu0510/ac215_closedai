from unittest.mock import patch


@patch('app.generate_query_embedding')
@patch('app.get_doc_from_client')
@patch('app.generate_response')
def test_chat_route(mock_generate_response, mock_get_doc_from_client, mock_generate_query_embedding, client):
    mock_generate_query_embedding.return_value = [0.1, 0.2, 0.3]
    mock_get_doc_from_client.return_value = {'documents': [["doc1", "doc2"]]}
    mock_generate_response.return_value = "This is a generated response."

    response = client.post('/chat', json={'message': 'Hello!'})
    data = response.get_json()

    assert response.status_code == 200
    assert 'reply' in data
    assert data['reply'] == "This is a generated response."


@patch('app.generate_query_embedding')
@patch('app.get_doc_from_client')
@patch('app.generate_response')
def test_no_input_route(mock_generate_response, mock_get_doc_from_client, mock_generate_query_embedding, client):
    mock_generate_query_embedding.return_value = [0.1, 0.2, 0.3]
    mock_get_doc_from_client.return_value = {'documents': [["doc1", "doc2"]]}
    mock_generate_response.return_value = "This is a generated response."

    response = client.post('/chat', json={'message': None})
    data = response.get_json()

    assert response.status_code == 200
    assert 'reply' in data
    assert data['reply'] == "Please enter a message."
