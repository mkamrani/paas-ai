# Core profile management

## Commands:
paas-ai config profiles                    # List all profiles
paas-ai config show                        # Show current config
paas-ai config show --profile <name>       # Show specific profile
paas-ai config set-current <profile>       # Switch active profile

# Profile creation/management  
paas-ai config add-profile <name> [options]  # Add custom profile
paas-ai config remove-profile <name>         # Remove custom profile
paas-ai config init                          # Initialize config file

# Utilities
paas-ai config validate                    # Validate current config
paas-ai config edit                        # Edit in $EDITOR

## Configuration Structure:
current: my-dev-profile
profiles:
  my-dev-profile:
    embedding:
      type: sentence_transformers
      model_name: all-MiniLM-L6-v2
    vectorstore:
      type: chroma
      persist_directory: ./rag_data/dev
      collection_name: dev-knowledge
    retriever:
      type: similarity
      search_kwargs: {k: 5}
    batch_size: 32
    validate_urls: true