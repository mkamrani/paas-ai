src/paas_ai/core/rag/
├── config.py                    # Configuration schemas
├── pipeline.py                  # Main RAG processor  
├── __init__.py                  # Module exports
├── README.md                    # Documentation
│
├── loaders/                     # 🔹 10 Document Loader Strategies
│   ├── __init__.py             
│   ├── base.py                 # Base strategy interface
│   ├── factory.py              # Strategy-based factory
│   ├── registry.py             # Strategy registry
│   ├── web.py                  # Web documents
│   ├── pdf.py                  # PDF files
│   ├── html.py                 # HTML files
│   ├── markdown.py             # Markdown files
│   ├── json.py                 # JSON documents
│   ├── csv.py                  # CSV files
│   ├── directory.py            # Local directories
│   ├── confluence.py           # Confluence pages
│   ├── notion.py               # Notion exports
│   └── github.py               # GitHub repositories
│
├── splitters/                   # 🔹 7 Text Splitter Strategies
│   ├── __init__.py
│   ├── base.py                 # Base strategy interface
│   ├── factory.py              # Strategy-based factory
│   ├── registry.py             # Strategy registry
│   ├── character.py            # Character-based splitting
│   ├── recursive_character.py  # Smart recursive splitting
│   ├── markdown.py             # Markdown-aware splitting
│   ├── html.py                 # HTML-aware splitting
│   ├── json.py                 # JSON structure-aware
│   ├── code.py                 # Programming language-aware
│   └── token.py                # Token-based splitting
│
├── embeddings/                  # 🔹 Embedding Strategies (expandable)
│   ├── __init__.py
│   ├── base.py                 # Base strategy interface
│   ├── factory.py              # Strategy-based factory
│   ├── registry.py             # Strategy registry
│   └── openai.py               # OpenAI embeddings (+ 4 more to add)
│
├── vectorstore/                 # 🔹 Vector Store Strategies (expandable)
│   ├── __init__.py
│   ├── base.py                 # Base strategy interface
│   ├── factory.py              # Strategy-based factory
│   └── chroma.py               # Chroma vector store (+ FAISS, Pinecone to add)
│
└── retrievers/                  # 🔹 Retriever Strategies (expandable)
    ├── __init__.py
    ├── base.py                 # Base strategy interface
    ├── factory.py              # Strategy-based factory
    └── similarity.py           # Similarity retrieval (+ 5 more to add)