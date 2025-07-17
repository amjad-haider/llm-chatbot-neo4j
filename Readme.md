# Setup

Create the virtual environment with `uv` or python `python venv` and install requirements.txt

Start Ollama

```bash
ollama serve
```

Download any models

```bash
ollama pull mistral:latest
```

Edit the variables in the `.env` file from `.env.example`


To start the chatbot, run:

```bash
streamlit run bot.py
```
