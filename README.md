# PaaS AI - Intelligent Platform as a Service Configuration Generator

🤖 **AI-powered PaaS configuration generation**

PaaS AI is my response to the need for a solution to streamline the process of deploying applications on custom/in-house built Platform as a Service (PaaS) systems.

Many times, the companies build their own Platform as a Service (PaaS) systems to deploy their applications. This, has multiple benefits and mainly is meant to make the life of everyone who needs to deploy their application in the environment easier. However, the main challenge is that the users now have to learn the DSL of the PaaS system, and even worse, they have to learn the best practices and guidelines specific to the PaaS system.

PaaS AI is a solution to this problem. It understands the requirements of the user, and after analyzing the requirements, designs the architecture of the application, and then generates the configurations for the application.

The tests, with a fictional PaaS system called Cool Demo PaaS, show the system is capable of generating descent configurations for hosting the applications. Even if it is not perfect, it is a really really good starting point for the users to get started.

While I have named the solution PaaS AI, it is not limited to PaaS systems. It can be used for any DSL (custom or existing like Kubernetes, and Terraform), ideally with the best practices and guidelines specific to your environment.

## 🚀 Quick Start

The fastest way to get started is with our interactive Jupyter notebook:

1. **Open the Getting Started Notebook**:
   ```bash
   jupyter notebook getting-started.ipynb
   ```

2. **Follow the step-by-step guide** that will:
   - Set up your environment
   - Create a custom configuration profile
   - Start the documentation server
   - Launch the AI chat interface

That's it! The notebook handles everything for you.

## 🎯 What Can PaaS AI Do?

- **Generate Configurations**: Create PaaS configs from natural language
- **Best Practices**: Get recommendations based on best practices and guidelines specific to the PaaS you are using
- **Interactive Chat**: Ask questions and get instant, contextual answers
- **Knowledge Base**: Access comprehensive PaaS documentation and examples

## 💬 Example Conversations

```
Q: "Help me with setting up my full stack application using Cool Demo PaaS."

Q: "What options do I have for hosting my React application using Cool Demo PaaS?"

Q: "How can I reduce the infrastructure cost of my application?"
```

## 🛠️ Manual Setup (Alternative)

If you prefer manual setup instead of the notebook:

### Prerequisites
- Python 3.11 or 3.12
- Poetry (for dependency management)
- Node.js 18+ (for documentation server)
- OpenAI API key

### Installation
```bash
# Install dependencies
poetry install --with rag,agent,api

# Set up environment variables
cp env.example .env
# Edit .env and add your OPENAI_API_KEY

# Initialize configuration
poetry run paas-ai config init

# Start the chat
poetry run paas-ai agent chat
```

## 📚 Documentation

- **Getting Started**: `getting-started.ipynb` - Interactive setup guide
- **Developer Guide**: `developers-guide.md` - Detailed technical documentation
- **Examples**: `examples/` - Sample configurations and use cases
- **API Documentation**: `docs/api/` - REST API reference (Coming soon)

## 🌐 Live Documentation

Start the documentation server to browse examples and guides:

```bash
cd examples/paas/docs
npm install
npm start
```

Visit http://localhost:3000 to explore PaaS configurations, best practices, and examples.

## 🎮 Available Commands

```bash
# Chat with the AI
poetry run paas-ai agent chat

# Manage configurations
poetry run paas-ai config show
poetry run paas-ai config create-profile --name my-profile

# Manage knowledge base
poetry run paas-ai rag index --source /path/to/docs
poetry run paas-ai rag search "PaaS deployment"

# Get help
poetry run paas-ai --help
```

## 🔧 Configuration Profiles

PaaS AI supports multiple configuration profiles for different use cases:

- **Default**: General-purpose configuration with OpenAI embeddings
- **Local**: Uses local sentence transformers for offline operation
- **Custom**: Create your own profiles with specific models and settings

## 🤝 Contributing

We welcome contributions! See `developers-guide.md` for detailed information about:

- Architecture and design
- Development setup
- Testing guidelines
- Contributing workflow

## 📄 License

MIT License - see `LICENSE` file for details.

## 🆘 Support

- **Issues**: Report bugs and request features on GitHub
- **Documentation**: Check `developers-guide.md` for detailed information
- **Examples**: Browse `examples/` for sample configurations

---

**Ready to get started?** Open `getting-started.ipynb` and let PaaS AI guide you through the setup! 🚀
