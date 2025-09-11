# Multi-Agent System Implementation Plan

## Overview

Replace the current single `RAGAgent` with a `MultiAgentSystem` that coordinates specialized agents in either **supervisor** or **swarm** mode, maintaining backward compatibility while providing enhanced capabilities.

## Agent Specializations

### 1. **Designer Agent**
- **Domain**: System architecture, design patterns, infrastructure planning
- **Capabilities**:
  - Analyze requirements and suggest system architectures
  - Generate Mermaid diagrams for system designs
  - Explain architectural patterns (microservices, serverless, event-driven)
  - Recommend technology stacks and design decisions
  - Apply environment constraints and rules (storage types, ports, etc.)

### 2. **K8s Manifest Agent**
- **Domain**: Kubernetes deployment, YAML generation, container orchestration
- **Capabilities**:
  - Generate Kubernetes manifest YAML files
  - Create deployment, service, ingress configurations
  - Generate ConfigMaps, Secrets, PersistentVolumes
  - Recommend best practices for K8s deployments
  - Troubleshoot deployment issues

## Architecture Components

### Core Components

```
src/paas_ai/core/agents/
├── __init__.py                    # Export MultiAgentSystem as main entry
├── multi_agent_system.py         # Main MAS coordinator (replaces RAGAgent)
├── base_agent.py                  # Base agent wrapper around create_react_agent
├── tool_registry.py               # Central tool registry for name-based lookup
│
├── tools/
│   ├── __init__.py
│   ├── rag_search.py              # Updated to use runtime config
│   ├── design_tools.py            # New: Mermaid, architecture patterns
│   ├── k8s_tools.py               # New: YAML generation, K8s utilities
│   └── handoff_tools.py           # Handoff tool factory functions
│
├── prompts/
│   ├── designer/
│   │   └── system.md              # Designer agent system prompt
│   ├── k8s_manifest/
│   │   └── system.md              # K8s agent system prompt
│   └── supervisor/
│       └── system.md              # Supervisor coordination prompt
│
└── new_design.md                  # This document
```

### Configuration Schema Updates

```python
class MultiAgentConfig(BaseModel):
    mode: Literal["supervisor", "swarm"] = "supervisor"
    default_agent: Literal["designer", "k8s_manifest"] = "designer"

class Config(BaseModel):
    # ... existing fields ...
    multi_agent: MultiAgentConfig = MultiAgentConfig()
    agents: Dict[str, Dict[str, Any]] = Field(
        default_factory=lambda: {
            "designer": {
                "model": "gpt-4o-mini",
                "temperature": 0.1
            },
            "k8s_manifest": {
                "model": "gpt-4o-mini", 
                "temperature": 0.0  # More deterministic for YAML generation
            }
        }
    )
```

## Implementation Plan

### Phase 1: Foundation (Base Classes and Registry)

#### 1.1 Create Tool Registry
- [ ] Implement `ToolRegistry` class with name-based lookup
- [ ] Create base tool classes that access config from `RunnableConfig`
- [ ] Update existing `RAGSearchTool` to use runtime config

#### 1.2 Create BaseAgent
- [ ] Implement `BaseAgent` wrapper around `create_react_agent`
- [ ] Support tool names instead of instances
- [ ] Automatic handoff tool generation for swarm mode
- [ ] Prompt template loading from `prompts/{agent_name}/system.md`

#### 1.3 Create HandoffTools
- [ ] Implement handoff tool factory using LangGraph patterns
- [ ] Support both supervisor and swarm handoff mechanisms

### Phase 2: Specialized Tools

#### 2.1 Design Tools
- [ ] `MermaidDiagramTool` - Generate architecture diagrams
- [ ] `ArchitecturePatternTool` - Explain and recommend patterns
- [ ] `TechnologyStackTool` - Suggest tech stacks based on requirements

#### 2.2 K8s Tools  
- [ ] `GenerateManifestTool` - Create K8s YAML files
- [ ] `K8sValidatorTool` - Validate manifest syntax and best practices
- [ ] `DeploymentTroubleshootTool` - Debug deployment issues

### Phase 3: Multi-Agent System

#### 3.1 Create MultiAgentSystem
- [ ] Main coordinator class that replaces `RAGAgent`
- [ ] Support supervisor mode using `langgraph-supervisor`
- [ ] Support swarm mode using `langgraph-swarm`
- [ ] Maintain identical API surface (`ask()`, `chat()`, etc.)

#### 3.2 Agent Definitions
- [ ] Create Designer agent with design-focused tools
- [ ] Create K8s Manifest agent with deployment tools
- [ ] Write specialized system prompts for each agent

### Phase 4: Integration

#### 4.1 Configuration Updates
- [ ] Extend config schema with multi-agent settings
- [ ] Update default profiles with multi-agent defaults
- [ ] Ensure backward compatibility

#### 4.2 CLI/API Integration
- [ ] Update imports to use `MultiAgentSystem`
- [ ] Ensure zero breaking changes to existing interfaces
- [ ] Test backward compatibility

#### 4.3 Prompt Templates
- [ ] Write Designer agent system prompt with architecture focus
- [ ] Write K8s Manifest agent system prompt with deployment focus
- [ ] Write Supervisor coordination prompt

### Phase 5: Testing and Validation

#### 5.1 Unit Tests
- [ ] Test tool registry and tool creation
- [ ] Test BaseAgent functionality
- [ ] Test MultiAgentSystem coordination

#### 5.2 Integration Tests
- [ ] Test supervisor mode coordination
- [ ] Test swarm mode handoffs
- [ ] Test backward compatibility with existing CLI/API

#### 5.3 End-to-End Scenarios
- [ ] "Design a microservices architecture" (Designer agent)
- [ ] "Generate K8s manifests for this design" (K8s agent)
- [ ] "Design and deploy a web application" (Cross-agent handoff)

## Tool Specifications

### Design Tools

#### MermaidDiagramTool
```python
name: "generate_mermaid_diagram"
description: "Generate Mermaid diagrams for system architecture visualization"
inputs: 
  - diagram_type: "flowchart" | "sequence" | "class" | "deployment"
  - components: List of system components
  - relationships: Component relationships
output: Mermaid markup text
```

#### ArchitecturePatternTool
```python
name: "explain_architecture_pattern"
description: "Explain architectural patterns and provide implementation guidance"
inputs:
  - pattern_name: Name of the pattern
  - context: Application context
output: Pattern explanation and implementation guide
```

### K8s Tools

#### GenerateManifestTool
```python
name: "generate_k8s_manifest"
description: "Generate Kubernetes YAML manifest files"
inputs:
  - resource_type: "deployment" | "service" | "ingress" | "configmap"
  - app_config: Application configuration
  - environment: Target environment settings
output: Complete K8s YAML manifest
```

#### K8sValidatorTool
```python
name: "validate_k8s_manifest"
description: "Validate Kubernetes manifests for syntax and best practices"
inputs:
  - manifest_yaml: YAML content to validate
output: Validation results and recommendations
```

## Backward Compatibility Strategy

1. **Same Interface**: `MultiAgentSystem` exposes identical methods to `RAGAgent`
2. **Import Alias**: `from .multi_agent_system import MultiAgentSystem as RAGAgent`
3. **Default Behavior**: Supervisor mode provides similar experience to single agent
4. **Configuration**: Multi-agent config gets sensible defaults if not specified
5. **Graceful Degradation**: System works even if specialized tools fail

## Success Criteria

- [ ] Existing CLI commands work without modification
- [ ] Existing API endpoints work without modification  
- [ ] Users can opt into swarm mode for dynamic agent coordination
- [ ] Specialized agents provide enhanced capabilities in their domains
- [ ] System maintains performance characteristics of single agent
- [ ] Tool registry enables easy addition of new capabilities

## Example Usage Scenarios

### Supervisor Mode (Default)
```bash
# User asks: "Design a microservices architecture for an e-commerce platform"
# Supervisor routes to Designer agent
# Designer uses RAG search + architecture tools to provide design

# User asks: "Generate K8s manifests for this design"  
# Supervisor routes to K8s Manifest agent
# K8s agent generates appropriate YAML files
```

### Swarm Mode
```bash
# User asks: "Design and deploy a scalable web application"
# Designer agent starts, creates architecture
# Designer hands off to K8s agent: "Here's the design, create deployment"
# K8s agent generates manifests based on design
# Seamless collaboration between specialized agents
```

## Next Steps

Start with Phase 1 implementation, focusing on the tool registry and base agent foundation that will support the entire system. 