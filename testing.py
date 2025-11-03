import requests
import sys
import os

def test_ocr_api(pdf_path="test.pdf", api_url="http://localhost:8080"):
    """
    Tests the OCR API by sending a PDF file and printing the response.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Please provide a valid PDF file path.")
        return False
    
    try:
        # Test health endpoint first
        print(f"Testing health endpoint: {api_url}/health")
        health_response = requests.get(f"{api_url}/health", timeout=10)
        print(f"Health Status: {health_response.status_code}")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"OCR Reader Loaded: {health_data.get('ocr_reader_loaded', False)}")
            print(f"Languages: {health_data.get('languages', [])}")
        print()
        
        # Test prediction endpoint
        print(f"Testing prediction endpoint with: {pdf_path}")
        print(f"File size: {os.path.getsize(pdf_path)} bytes")
        
        with open(pdf_path, "rb") as f:
            files = {"file": (os.path.basename(pdf_path), f, "application/pdf")}
            response = requests.post(
                f"{api_url}/predict", 
                files=files,
                timeout=120  # 2 minute timeout for processing
            )
        
        print(f"Response Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("SUCCESS! Extraction Results:")
            print("="*60)
            print(f"Pages processed: {result.get('pages_processed', 'N/A')}")
            print(f"Status: {result.get('status', 'N/A')}")
            print("\nExtracted Text:")
            print("="*60)
            print(result.get("text", ""))
            print("="*60)
            return True
        else:
            error_data = response.json()
            print(f"\nError: {error_data.get('error', 'Unknown error')}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API server at {api_url}")
        print("Make sure the server is running with: python app.py")
        return False
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The PDF might be too large or complex.")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "test.pdf"
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"
    
    success = test_ocr_api(pdf_file, api_url)
    sys.exit(0 if success else 1)