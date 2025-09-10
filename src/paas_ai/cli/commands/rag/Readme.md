# RAG Commands Organization

This directory organizes RAG CLI commands into focused modules by functionality.

## Structure

```
rag/
├── __init__.py      # Main rag group + subcommand registration
├── resources.py     # Resource management (list, add, remove)
├── sync.py         # Synchronization operations
├── status.py       # System status and health
├── search.py       # Knowledge base search
└── reports.py      # Analytics and reporting
```

## Usage

All commands are accessed through the `paas-ai rag` namespace:

```bash
paas-ai rag resources list
paas-ai rag sync --incremental
paas-ai rag status --detailed
paas-ai rag search "kubernetes"
paas-ai rag report sync-history
```

## Adding New Commands

1. Create new `.py` file with your command(s)
2. Import and register in `__init__.py`
3. Follow existing patterns for consistency