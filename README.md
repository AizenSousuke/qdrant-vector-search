# Qdrant search with FastAPI

This is a simple FastAPI application that uses Qdrant to store and search vectors.

## Setup
You need a few items to get started:
- Docker image of Qdrant
- Python 3.13 or higher with uv
- A virtual environment

### Docker image of Qdrant
You can use the official Qdrant docker image, or build your own.
```
docker pull qdrant/qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Python 3.13 or higher
You can use any Python version you like, but this example uses Python 3.13.

### Virtual environment
Create a virtual environment and install the dependencies:
```
uv sync
```

## Running the application
To run the application, use uvicorn:
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Testing the application
To test the application, you can use curl or Postman.
You can also use vscode with curl runner extension to run curl commands from main.py to see how this application works.

## Contributing
If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.