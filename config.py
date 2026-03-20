# Ollama runs locally — no API key needed
OLLAMA_BASE_URL = "http://localhost:11434"

# Model to use — must be pulled first via: ollama pull <model>
# options: llama3.2, llama3:latest, phi3:mini
MODEL_NAME = "llama3:latest"

# Agent settings
MAX_ITERATIONS = 10
VERBOSE = True