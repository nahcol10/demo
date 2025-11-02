import requests

with open("test_file.pdf", "rb") as f:
    response = requests.post("http://localhost:8080/predict", files={"file": f})
    print(response.json())