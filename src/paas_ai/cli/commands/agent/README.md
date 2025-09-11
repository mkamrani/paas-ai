# Agent Commands

This module provides CLI commands for interacting with the RAG (Retrieval-Augmented Generation) agent.

## Commands

### `paas-ai agent ask`

Ask the RAG agent a single question. This is useful for one-off queries where you don't need conversation context.

**Usage:**
```bash
# Basic question
paas-ai agent ask -q "What is Alice's job?"

# Show configuration summary
paas-ai agent ask -q "What features are available?" --show-config

# Use specific config profile
paas-ai agent ask -q "Tell me about the system" --config-profile local
```

**Options:**
- `-q, --question TEXT` - The question to ask (required)
- `--config-profile TEXT` - Override config profile for this operation
- `--show-config` - Show configuration summary before processing

### `paas-ai agent chat`

Start an interactive chat session with the RAG agent. This maintains conversation history and allows for follow-up questions and context-aware responses.

**Usage:**
```bash
# Start basic chat session
paas-ai agent chat

# Start with configuration display
paas-ai agent chat --show-config

# Limit conversation history to 10 exchanges
paas-ai agent chat --max-history 10
```

**Options:**
- `--config-profile TEXT` - Override config profile for this operation
- `--show-config` - Show configuration summary at startup
- `--max-history INTEGER` - Maximum number of messages to keep in history (default: 20)

**Interactive Commands:**
During a chat session, you can use these special commands:

- `history` - View conversation history
- `clear` - Clear conversation history and start fresh
- `tools` - Show available agent tools
- `config` - Show current configuration
- `exit`, `quit`, or `bye` - End the chat session

## Module Structure

```
agent/
├── __init__.py          # Module initialization and command group registration
├── ask.py              # Single question command implementation
├── chat.py             # Interactive chat command implementation
└── README.md           # This documentation file
```

## Key Features

- **Context Awareness**: The chat command maintains conversation history for context-aware responses
- **Configuration Display**: Both commands can show current configuration settings
- **Error Handling**: Graceful error handling with helpful error messages
- **Colorful Output**: Enhanced user experience with styled terminal output
- **Conversation Management**: Chat sessions support history viewing, clearing, and trimming
- **Tool Discovery**: Users can explore available agent capabilities

## Examples

### Single Question
```bash
$ paas-ai agent ask -q "What does Alice do for work?"
============================================================
AGENT RESPONSE:
============================================================
Alice is a curious software engineer who works at QuantumTech Industries.
============================================================
```

### Interactive Chat
```bash
$ paas-ai agent chat
============================================================
🤖 RAG AGENT INTERACTIVE CHAT SESSION
============================================================
💡 Commands:
  • 'exit' or 'quit' - End the session
  • 'clear' - Clear conversation history
  • 'history' - Show conversation history
  • 'tools' - Show available tools
  • 'config' - Show current configuration
============================================================
📝 Max history: 20 messages
============================================================

You: Tell me about Alice
🤔 Agent is thinking...

🤖 Agent: Alice is a curious software engineer who works at QuantumTech Industries...

You: What adventure did she go on?
🤔 Agent is thinking...

🤖 Agent: Alice went on a digital adventure where she tumbled through a virtual portal...

You: exit

👋 Thanks for chatting! Goodbye!
📊 Session completed: 2 exchanges, 4 total messages
```

src/paas_ai/cli/commands/agent/
├── __init__.py          # Module initialization and command group registration
├── ask.py              # Single question command implementation
├── chat.py             # Interactive chat command implementation
└── README.md           # Documentation for the agent commands