python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start Ollama First
ollama serve
ollama run llama3.2