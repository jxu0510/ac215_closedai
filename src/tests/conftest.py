import pytest
import sys
import os
sys.path.insert(0, os.path.abspath("./service"))
sys.path.insert(0, os.path.abspath("./finetune_data"))
sys.path.insert(0, os.path.abspath("./finetune_model"))
sys.path.insert(0, os.path.abspath("./rag_data_pipeline"))
from app import app 

@pytest.fixture
def client():
    """Fixture for Flask test client."""
    with app.test_client() as client:
        yield client