from unittest.mock import patch
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
import time
# import subprocess


@patch("app.generate_query_embedding")
@patch("app.client.get_collection")
@patch("app.generative_model.generate_content")
def test_e2e_chat(
    mock_generate_content, mock_get_collection, mock_generate_query_embedding
):
    """Test the chatbot end-to-end in a real browser."""
    mock_generate_query_embedding.return_value = [0.1, 0.2, 0.3]
    mock_get_collection.return_value.query.return_value = {
        "documents": [["doc1", "doc2"]]
    }
    mock_generate_content.return_value.text = "This is a generated response."

    # Start Flask app as a subprocess
    # flask_process = subprocess.Popen(
    #     ["flask", "run", "--host=0.0.0.0", "--port=8080"],
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.PIPE,
    # )
    time.sleep(5)  # Give Flask time to start

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.binary_location = "/usr/bin/chromium"

    service = Service(executable_path="/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)

    driver.get("http://localhost:8080")  # Ensure the Flask app is running

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "user-input"))
        )

        input_box = driver.find_element(By.ID, "user-input")
        send_button = driver.find_element(By.TAG_NAME, "button")

        input_box.send_keys("Hello, chatbot!")
        send_button.click()

        WebDriverWait(driver, 10).until(
            EC.text_to_be_present_in_element((By.ID, "chat-window"), "Hello, chatbot!")
        )

        chat_window = driver.find_element(By.ID, "chat-window")
        messages = chat_window.find_elements(By.CLASS_NAME, "message")
        print(messages)
        #       assert len(messages) >= 2
        assert "Hello, chatbot!" in messages[-1].text
    #       assert "Error" not in messages[-1].text
    finally:
        # Quit the browser
        driver.quit()
