#!/usr/bin/env python
# coding: utf-8

import requests
import sys
import os

def test_ocr_api(pdf_path="jpt.pdf", api_url="http://localhost:8080"):
    """
    Tests the OCR API by sending a PDF file and printing the response.
    
    Args:
        pdf_path: Path to the PDF file to test
        api_url: Base URL of the API server
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found.")
        print("Usage: python testing.py [path_to_pdf_file] [api_url]")
        sys.exit(1)
    
    try:
        # Test health endpoint first
        print(f"Testing health endpoint: {api_url}/health")
        health_response = requests.get(f"{api_url}/health", timeout=5)
        print(f"Health Status: {health_response.status_code}")
        print(f"Health Response: {health_response.json()}\n")
        
        if health_response.status_code != 200:
            print("Warning: API health check failed. Continuing anyway...")
        
        # Test prediction endpoint
        print(f"Testing prediction endpoint: {api_url}/predict")
        print(f"Sending PDF file: {pdf_path}")
        
        with open(pdf_path, "rb") as f:
            response = requests.post(
                f"{api_url}/predict", 
                files={"file": f},
                timeout=60  # 60 second timeout for processing
            )
        
        print(f"\nResponse Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "="*60)
            print("SUCCESS! Extracted Text:")
            print("="*60)
            print(result.get("text", ""))
            print("="*60)
        else:
            print(f"\nError Response: {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API server at {api_url}")
        print("Make sure the server is running with: python app.py")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The PDF might be too large or complex.")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "test_file.pdf"
    api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"
    
    test_ocr_api(pdf_file, api_url)