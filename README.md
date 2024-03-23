# RAG-chat-with-documents
Chainlit app for advanced RAG. Uses llamaparse, langchain, qdrant and GPT4.


## Videos covering these topics
### [Llamaparse LlamaIndex](https://youtu.be/wRMnHbiz5ck?si=iQZV7N6-trcuBm8M)
### [RAG With LlamaParse from LlamaIndex & LangChain 🚀](https://youtu.be/f9hvrqVvZl0?si=qnJBsAZD4hBUweiS)

### Links shown in video
- [LlamaCloud](https://cloud.llamaindex.ai/)
- [Qdrant Cloud](https://cloud.qdrant.io/)

### create virtualenv
```
python3 -m venv .venv && source .venv/bin/activate
```

### Install packages
```
pip install -r requirements.txt
```

### Environment variables
All env variables goes to .env ( cp to `.env` and paste required env variables)

### Run the python files (following the video to run step by step is recommended)
```
python3 ingest.py
chainlit run app.py
```

