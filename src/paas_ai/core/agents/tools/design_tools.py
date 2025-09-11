"""
Design tools for the Designer agent.

Provides tools for architecture design, Mermaid diagram generation, and pattern explanations.
"""

from typing import Any, Dict, Optional, Literal
from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from ....utils.logging import get_logger

logger = get_logger("paas_ai.agents.design_tools")


class MermaidDiagramInput(BaseModel):
    """Input schema for Mermaid diagram generation."""
    diagram_type: Literal["flowchart", "sequence", "class", "deployment", "er", "state"] = Field(
        description="Type of diagram to generate"
    )
    title: str = Field(description="Title for the diagram")
    components: str = Field(description="Description of system components and their roles")
    relationships: str = Field(description="Description of relationships between components")
    additional_context: Optional[str] = Field(
        default=None, 
        description="Additional context or constraints for the diagram"
    )


class MermaidDiagramTool(BaseTool):
    """Tool for generating Mermaid diagrams for system architecture visualization."""
    
    name: str = "generate_mermaid_diagram"
    description: str = """
    Generate Mermaid diagrams for system architecture visualization.
    
    Use this tool to create visual representations of:
    - System architectures (flowchart)
    - Sequence diagrams for process flows
    - Class diagrams for object relationships
    - Deployment diagrams for infrastructure
    - Entity-relationship diagrams for data models
    - State diagrams for process states
    
    Provide clear descriptions of components and their relationships.
    """
    args_schema: type = MermaidDiagramInput
    
    def __init__(self):
        """Initialize the Mermaid diagram tool."""
        super().__init__()
    
    def _run(
        self,
        diagram_type: str,
        title: str,
        components: str,
        relationships: str,
        additional_context: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Generate a Mermaid diagram based on the provided specifications."""
        try:
            # Create diagram based on type
            if diagram_type == "flowchart":
                return self._create_flowchart(title, components, relationships, additional_context)
            elif diagram_type == "sequence":
                return self._create_sequence_diagram(title, components, relationships, additional_context)
            elif diagram_type == "class":
                return self._create_class_diagram(title, components, relationships, additional_context)
            elif diagram_type == "deployment":
                return self._create_deployment_diagram(title, components, relationships, additional_context)
            elif diagram_type == "er":
                return self._create_er_diagram(title, components, relationships, additional_context)
            elif diagram_type == "state":
                return self._create_state_diagram(title, components, relationships, additional_context)
            else:
                return f"Unsupported diagram type: {diagram_type}"
                
        except Exception as e:
            logger.error(f"Error generating Mermaid diagram: {e}")
            return f"Error generating diagram: {str(e)}"
    
    def _create_flowchart(self, title: str, components: str, relationships: str, context: Optional[str] = None) -> str:
        """Create a flowchart diagram."""
        return f"""# {title}

```mermaid
flowchart TD
    %% {title}
    
    %% Components based on: {components}
    %% Relationships: {relationships}
    {f"%% Additional context: {context}" if context else ""}
    
    A[User/Client] --> B[Load Balancer]
    B --> C[API Gateway]
    C --> D[Authentication Service]
    C --> E[Business Logic Services]
    E --> F[Database Layer]
    E --> G[Cache Layer]
    E --> H[Message Queue]
    
    %% Styling
    classDef userClass fill:#e1f5fe
    classDef serviceClass fill:#f3e5f5
    classDef dataClass fill:#e8f5e8
    
    class A userClass
    class B,C,D,E serviceClass
    class F,G,H dataClass
```

**Generated Architecture Diagram**

This flowchart represents: {components}

**Key Relationships**: {relationships}

{f"**Additional Considerations**: {context}" if context else ""}

**Diagram Elements**:
- **Blue**: User-facing components
- **Purple**: Service layer components  
- **Green**: Data and infrastructure components
"""

    def _create_sequence_diagram(self, title: str, components: str, relationships: str, context: Optional[str] = None) -> str:
        """Create a sequence diagram."""
        return f"""# {title}

```mermaid
sequenceDiagram
    title {title}
    
    %% Participants based on: {components}
    participant Client
    participant API
    participant Auth
    participant Service
    participant DB
    
    %% Flow based on: {relationships}
    Client->>+API: Request
    API->>+Auth: Validate Token
    Auth-->>-API: Token Valid
    API->>+Service: Process Request
    Service->>+DB: Query Data
    DB-->>-Service: Return Data
    Service-->>-API: Response Data
    API-->>-Client: Final Response
    
    Note over Client,DB: {relationships}
```

**Generated Sequence Diagram**

This sequence diagram shows: {components}

**Process Flow**: {relationships}

{f"**Context**: {context}" if context else ""}
"""

    def _create_deployment_diagram(self, title: str, components: str, relationships: str, context: Optional[str] = None) -> str:
        """Create a deployment/infrastructure diagram."""
        return f"""# {title}

```mermaid
flowchart TB
    subgraph "Production Environment"
        subgraph "Load Balancing"
            LB[Load Balancer]
        end
        
        subgraph "Application Tier"
            A1[App Instance 1]
            A2[App Instance 2]
            A3[App Instance 3]
        end
        
        subgraph "Data Tier"
            DB[(Primary Database)]
            CACHE[(Redis Cache)]
            Q[Message Queue]
        end
        
        subgraph "Monitoring"
            MON[Monitoring Service]
            LOG[Logging Service]
        end
    end
    
    Internet --> LB
    LB --> A1
    LB --> A2
    LB --> A3
    
    A1 --> DB
    A2 --> DB
    A3 --> DB
    
    A1 --> CACHE
    A2 --> CACHE
    A3 --> CACHE
    
    A1 --> Q
    A2 --> Q
    A3 --> Q
    
    A1 --> MON
    A2 --> MON
    A3 --> MON
    
    A1 --> LOG
    A2 --> LOG
    A3 --> LOG
```

**Generated Deployment Diagram**

**Infrastructure Components**: {components}

**Deployment Architecture**: {relationships}

{f"**Additional Deployment Notes**: {context}" if context else ""}
"""

    def _create_class_diagram(self, title: str, components: str, relationships: str, context: Optional[str] = None) -> str:
        """Create a class diagram."""
        return f"""# {title}

```mermaid
classDiagram
    title {title}
    
    class User {{
        +String id
        +String email
        +String name
        +login()
        +logout()
    }}
    
    class Service {{
        +String id
        +String name
        +process()
        +validate()
    }}
    
    class Repository {{
        +find()
        +save()
        +delete()
    }}
    
    User --> Service : uses
    Service --> Repository : accesses
    
    note for User "Represents system users"
    note for Service "Business logic layer"
    note for Repository "Data access layer"
```

**Generated Class Diagram**

**System Components**: {components}

**Class Relationships**: {relationships}

{f"**Design Notes**: {context}" if context else ""}
"""

    def _create_er_diagram(self, title: str, components: str, relationships: str, context: Optional[str] = None) -> str:
        """Create an entity-relationship diagram."""
        return f"""# {title}

```mermaid
erDiagram
    USER ||--o{{ ORDER : places
    USER {{
        string id PK
        string email
        string name
        datetime created_at
    }}
    
    ORDER ||--|{{ ORDER_ITEM : contains
    ORDER {{
        string id PK
        string user_id FK
        decimal total
        datetime created_at
        string status
    }}
    
    ORDER_ITEM }}o--|| PRODUCT : references
    ORDER_ITEM {{
        string id PK
        string order_id FK
        string product_id FK
        int quantity
        decimal price
    }}
    
    PRODUCT {{
        string id PK
        string name
        string description
        decimal price
        int stock
    }}
```

**Generated Entity-Relationship Diagram**

**Data Entities**: {components}

**Entity Relationships**: {relationships}

{f"**Data Model Notes**: {context}" if context else ""}
"""

    def _create_state_diagram(self, title: str, components: str, relationships: str, context: Optional[str] = None) -> str:
        """Create a state diagram."""
        return f"""# {title}

```mermaid
stateDiagram-v2
    [*] --> Initial
    Initial --> Processing : start
    Processing --> Success : complete
    Processing --> Failed : error
    Processing --> Cancelled : cancel
    
    Success --> [*]
    Failed --> Retry : retry
    Failed --> [*] : abandon
    Cancelled --> [*]
    
    Retry --> Processing : restart
    
    note right of Processing
        {relationships}
    end note
```

**Generated State Diagram**

**System States**: {components}

**State Transitions**: {relationships}

{f"**Process Notes**: {context}" if context else ""}
"""


class ArchitecturePatternInput(BaseModel):
    """Input schema for architecture pattern explanation."""
    pattern_name: str = Field(description="Name of the architectural pattern to explain")
    context: Optional[str] = Field(
        default=None,
        description="Specific context or use case for the pattern"
    )
    constraints: Optional[str] = Field(
        default=None,
        description="Any constraints or requirements to consider"
    )


class ArchitecturePatternTool(BaseTool):
    """Tool for explaining architectural patterns and providing implementation guidance."""
    
    name: str = "explain_architecture_pattern"
    description: str = """
    Explain architectural patterns and provide detailed implementation guidance.
    
    Use this tool to get comprehensive explanations of:
    - Microservices architecture
    - Event-driven architecture
    - Serverless patterns
    - CQRS and Event Sourcing
    - Hexagonal architecture
    - Layered architecture
    - And many other patterns
    
    Provide the pattern name and any specific context for tailored guidance.
    """
    args_schema: type = ArchitecturePatternInput
    
    def __init__(self):
        """Initialize the architecture pattern tool."""
        super().__init__()
    
    def _run(
        self,
        pattern_name: str,
        context: Optional[str] = None,
        constraints: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs
    ) -> str:
        """Explain an architectural pattern with implementation guidance."""
        try:
            pattern_lower = pattern_name.lower()
            
            # Common patterns
            if "microservice" in pattern_lower:
                return self._explain_microservices(context, constraints)
            elif "event" in pattern_lower and "driven" in pattern_lower:
                return self._explain_event_driven(context, constraints)
            elif "serverless" in pattern_lower:
                return self._explain_serverless(context, constraints)
            elif "cqrs" in pattern_lower:
                return self._explain_cqrs(context, constraints)
            elif "hexagonal" in pattern_lower or "ports" in pattern_lower:
                return self._explain_hexagonal(context, constraints)
            elif "layered" in pattern_lower or "n-tier" in pattern_lower:
                return self._explain_layered(context, constraints)
            elif "mvc" in pattern_lower:
                return self._explain_mvc(context, constraints)
            elif "saga" in pattern_lower:
                return self._explain_saga(context, constraints)
            else:
                return self._explain_generic_pattern(pattern_name, context, constraints)
                
        except Exception as e:
            logger.error(f"Error explaining architecture pattern: {e}")
            return f"Error explaining pattern: {str(e)}"
    
    def _explain_microservices(self, context: Optional[str], constraints: Optional[str]) -> str:
        """Explain microservices architecture pattern."""
        return f"""# Microservices Architecture Pattern

## Overview
Microservices architecture is a method of developing software systems as a suite of independently deployable, small, modular services. Each service runs its own process and communicates via well-defined, lightweight mechanisms.

## Key Characteristics
- **Decentralized**: Services manage their own data and business logic
- **Independent Deployment**: Each service can be deployed independently
- **Technology Diversity**: Different services can use different technologies
- **Fault Isolation**: Failure in one service doesn't bring down the entire system
- **Scalability**: Scale individual services based on demand

## Implementation Approach

### 1. Service Design
- Single Responsibility: Each service has one business capability
- Database per Service: Each service manages its own data
- API-First: Well-defined interfaces between services

### 2. Communication Patterns
- **Synchronous**: HTTP/REST for real-time communication
- **Asynchronous**: Message queues for event-driven communication
- **API Gateway**: Single entry point for client requests

### 3. Data Management
- Database per service pattern
- Event sourcing for data consistency
- CQRS for read/write separation

## Benefits
- **Scalability**: Scale services independently
- **Technology Freedom**: Choose best tech for each service
- **Team Independence**: Teams can work independently
- **Fault Tolerance**: Isolated failures

## Challenges
- **Complexity**: Distributed system complexity
- **Data Consistency**: Managing transactions across services
- **Network Latency**: Inter-service communication overhead
- **Monitoring**: Distributed tracing and logging

## Best Practices
1. Start with a monolith, then extract services
2. Design for failure (circuit breakers, retries)
3. Implement comprehensive monitoring
4. Use containerization (Docker, Kubernetes)
5. Implement service discovery
6. Use API versioning strategies

{f"## Context-Specific Considerations\n{context}" if context else ""}
{f"## Constraint Analysis\n{constraints}" if constraints else ""}

## Recommended Tools & Technologies
- **Containers**: Docker, Kubernetes
- **API Gateway**: Kong, Zuul, Ambassador
- **Service Mesh**: Istio, Linkerd
- **Monitoring**: Prometheus, Grafana, Jaeger
- **Message Queues**: RabbitMQ, Apache Kafka
"""

    def _explain_event_driven(self, context: Optional[str], constraints: Optional[str]) -> str:
        """Explain event-driven architecture pattern."""
        return f"""# Event-Driven Architecture Pattern

## Overview
Event-driven architecture (EDA) is a software architecture pattern where the flow of the program is determined by events such as user actions, sensor outputs, or message passing from other programs.

## Core Components
- **Event Producers**: Generate events when something happens
- **Event Routers**: Route events to appropriate consumers
- **Event Consumers**: React to events and perform actions
- **Event Store**: Persistent storage for events

## Event Types
1. **Domain Events**: Business-significant occurrences
2. **Integration Events**: Cross-boundary communication
3. **System Events**: Infrastructure and operational events

## Implementation Patterns

### 1. Event Sourcing
- Store events as the primary source of truth
- Rebuild state by replaying events
- Immutable event log

### 2. CQRS (Command Query Responsibility Segregation)
- Separate read and write models
- Commands change state, queries read state
- Often combined with event sourcing

### 3. Saga Pattern
- Manage distributed transactions
- Compensating actions for failure scenarios
- Choreography vs Orchestration

## Benefits
- **Loose Coupling**: Components don't need direct knowledge of each other
- **Scalability**: Asynchronous processing enables better scaling
- **Flexibility**: Easy to add new event consumers
- **Auditability**: Complete event history

## Challenges
- **Complexity**: Understanding event flows
- **Eventual Consistency**: Data may be temporarily inconsistent
- **Event Schema Evolution**: Managing changes over time
- **Debugging**: Tracing issues across event flows

## Best Practices
1. Design events for business domain
2. Make events immutable
3. Include sufficient context in events
4. Handle duplicate events (idempotency)
5. Implement proper error handling
6. Use event versioning strategies

{f"## Context-Specific Considerations\n{context}" if context else ""}
{f"## Constraint Analysis\n{constraints}" if constraints else ""}

## Technology Stack
- **Message Brokers**: Apache Kafka, RabbitMQ, AWS SNS/SQS
- **Event Stores**: EventStore, Apache Kafka, AWS EventBridge
- **Stream Processing**: Apache Kafka Streams, Apache Flink
- **Monitoring**: Event tracking, message tracing
"""

    def _explain_serverless(self, context: Optional[str], constraints: Optional[str]) -> str:
        """Explain serverless architecture pattern."""
        return f"""# Serverless Architecture Pattern

## Overview
Serverless computing is a cloud execution model where the cloud provider manages the infrastructure, automatically allocating resources as needed. You only pay for the actual compute time consumed.

## Core Principles
- **No Server Management**: Cloud provider handles infrastructure
- **Event-Driven**: Functions triggered by events
- **Automatic Scaling**: Scales automatically with demand
- **Pay-per-Use**: Only pay for actual execution time

## Serverless Components

### 1. Functions as a Service (FaaS)
- AWS Lambda, Azure Functions, Google Cloud Functions
- Stateless, short-lived functions
- Event-triggered execution

### 2. Backend as a Service (BaaS)
- Managed databases, authentication, storage
- Third-party services for common functionality

### 3. API Gateway
- Manage and route API requests
- Handle authentication, rate limiting, caching

## Architecture Patterns

### 1. Event-Driven Functions
```
Event Source → Function → Action
```

### 2. API-Based Functions
```
Client → API Gateway → Function → Database
```

### 3. Stream Processing
```
Data Stream → Processing Function → Output
```

## Benefits
- **Cost Efficiency**: Pay only for actual usage
- **Automatic Scaling**: Handle traffic spikes automatically
- **Reduced Operations**: No server management
- **Fast Development**: Focus on business logic

## Challenges
- **Cold Starts**: Initial latency for function execution
- **Vendor Lock-in**: Tied to specific cloud provider
- **Limited Execution Time**: Functions have time limits
- **Debugging Complexity**: Distributed debugging challenges

## Best Practices
1. Keep functions small and focused
2. Minimize cold start impact
3. Use appropriate triggers and events
4. Implement proper error handling
5. Monitor function performance
6. Design for statelessness

{f"## Context-Specific Considerations\n{context}" if context else ""}
{f"## Constraint Analysis\n{constraints}" if constraints else ""}

## Technology Recommendations
- **AWS**: Lambda, API Gateway, DynamoDB, S3
- **Azure**: Functions, API Management, Cosmos DB
- **Google Cloud**: Cloud Functions, Cloud Endpoints, Firestore
- **Monitoring**: CloudWatch, Application Insights, Stackdriver
"""

    def _explain_generic_pattern(self, pattern_name: str, context: Optional[str], constraints: Optional[str]) -> str:
        """Provide a generic explanation framework for any pattern."""
        return f"""# {pattern_name} Architecture Pattern

## Pattern Analysis

I don't have a specific template for "{pattern_name}", but I can provide a general analysis framework:

## Key Questions to Consider:
1. **What problem does this pattern solve?**
2. **What are the core components and their responsibilities?**
3. **How do components interact with each other?**
4. **What are the benefits and trade-offs?**
5. **When should you use this pattern?**
6. **What are common implementation approaches?**

## General Pattern Evaluation Framework:

### Benefits Analysis
- Performance characteristics
- Scalability implications
- Maintainability aspects
- Development complexity

### Trade-offs Consideration
- Implementation complexity vs benefits
- Performance vs flexibility
- Consistency vs availability
- Cost vs capabilities

### Implementation Guidance
- Start with simple implementation
- Identify key components
- Define clear interfaces
- Plan for testing and monitoring

{f"## Context-Specific Analysis\n{context}" if context else ""}
{f"## Constraint Considerations\n{constraints}" if constraints else ""}

## Recommendation
For specific guidance on "{pattern_name}", I recommend:
1. Research the pattern's official documentation
2. Look for implementation examples in your technology stack
3. Consider similar patterns that might be applicable
4. Evaluate if existing patterns can solve your use case

Would you like me to explain a specific aspect of this pattern or suggest alternative patterns that might fit your needs?
""" 