import pytest
import json
# Correctly import both the app and the initialization function
from api.building_companion_app import app as flask_app, initialize_app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        # Initialize the app context to ensure the agent is loaded
        with flask_app.app_context():
            # Call the initialize_app function directly
            initialize_app()
        yield client

def test_health_check(client):
    """Test the /health endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['agent_status'] == 'loaded'

def test_query_endpoint_success(client):
    """Test the /query endpoint with a valid question."""
    test_query = {"query": "Qual é a altura mínima para uma janela?"}
    response = client.post('/query', data=json.dumps(test_query), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'answer' in data
    assert isinstance(data['answer'], str) and len(data['answer']) > 0

def test_query_endpoint_missing_field(client):
    """Test the /query endpoint with a missing 'query' field."""
    response = client.post('/query', data=json.dumps({}), content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "Missing 'query' field" in data['error']

def test_query_endpoint_empty_query(client):
    """Test the /query endpoint with an empty query."""
    response = client.post('/query', data=json.dumps({"query": "  "}), content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "Query cannot be empty" in data['error']

def test_chat_interface_loads(client):
    """Test the root URL ('/') returns the HTML chat interface."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Building Companion" in response.data