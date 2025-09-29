# Citation System

A citation system that extends the RAG pipeline to provide precise source references, supporting different verbosity levels and resource-specific strategies.

## Overview

The citation system seamlessly integrates with your existing RAG architecture to add:

- **Precise Source Tracking**: Document-level to sentence-level citation granularity
- **Resource-Aware Strategies**: Different citation formats for different content types (APIs, policies, web content)
- **Configurable Verbosity**: From minimal source names to forensic-level quotes with line numbers
- **Deep Linking**: Automatic generation of clickable links to specific sections
- **Backward Compatibility**: Easily enabled/disabled without breaking existing functionality

## Features

### Verbosity Levels

- **NONE**: No citations (maintains existing behavior)
- **MINIMAL**: Just source name `[Kubernetes Docs]`
- **STANDARD**: Source + location `[Kubernetes Docs, Pod Configuration]`
- **DETAILED**: Full reference `[Kubernetes Documentation, Pod Configuration Guide, Section 'Resource Limits']`
- **FORENSIC**: Exact quotes + precise location `[Kubernetes Docs, Pod Configuration]: "You can specify resource requests and limits"`

### Resource-Specific Strategies

#### Technical Documentation (DSL)
```
MINIMAL:   [Kubernetes Docs]
STANDARD:  [Kubernetes Docs, Pod Configuration]
DETAILED:  [Kubernetes Docs, API Documentation, Section: Pod Configuration]
FORENSIC:  [Kubernetes Docs, Section: Pod Configuration]: "spec.containers[].resources"
```

#### Web Content (CONTEXTUAL)
```
MINIMAL:   [kubernetes.io]
STANDARD:  [kubernetes.io, Pod Configuration]
DETAILED:  [kubernetes.io/docs/concepts/workloads/pods, Pod Configuration Guide]
FORENSIC:  [kubernetes.io/docs/concepts/workloads/pods#resource-limits]: "Set resource requests and limits"
```

#### Policies (GUIDELINES)
```
MINIMAL:   [Security Policy]
STANDARD:  [Security Policy v3.2, Section 4]
DETAILED:  [Security Policy v3.2, Section 4.3 'Access Control Requirements', Page 15]
FORENSIC:  [Security Policy v3.2, Section 4.3.2, Requirement SC-4.3.2-01]: "All user accounts must implement MFA"
```

## Configuration

### Basic Setup

```python
from paas_ai.core.config.schemas import CitationConfig, CitationVerbosity

citation_config = CitationConfig(
    enabled=True,
    verbosity=CitationVerbosity.STANDARD,
    format=CitationFormat.INLINE
)
```

### Advanced Configuration

```python
citation_config = CitationConfig(
    enabled=True,
    verbosity=CitationVerbosity.DETAILED,
    format=CitationFormat.INLINE,
    
    # Resource-specific overrides
    resource_overrides={
        ResourceType.DSL: CitationVerbosity.FORENSIC,
        ResourceType.GUIDELINES: CitationVerbosity.DETAILED,
        ResourceType.CONTEXTUAL: CitationVerbosity.STANDARD
    },
    
    # Content preferences
    include_quotes=True,
    max_quote_length=150,
    include_confidence=True,
    generate_deep_links=True,
    
    # Strategy mapping
    strategies={
        ResourceType.DSL: "technical_citation",
        ResourceType.CONTEXTUAL: "web_citation",
        ResourceType.GUIDELINES: "policy_citation"
    }
)
```

### Profile-Based Configuration

Use pre-configured profiles for common scenarios:

```bash
# Standard citations for development
paas-ai rag search "kubernetes deployment" --config-profile default

# Detailed citations for production
paas-ai rag search "security policies" --config-profile production

# Forensic-level citations for compliance (using custom profile)
paas-ai rag search "compliance requirements" --config-profile my_citation_profile
```

## Integration with Existing System

### Backward Compatibility

The citation system is designed for zero-impact integration:

- **Default Behavior**: Set `citation.enabled=False` or `verbosity=NONE`
- **Existing Data**: Works with already-ingested documents
- **Gradual Migration**: Enable citations per profile or resource type

### CLI Integration

```bash
# Add resources with citations enabled
paas-ai rag resources add --url "https://kubernetes.io/docs" --type dsl

# Search with citations
paas-ai rag search "pod configuration"

# Results now include citation information:
# Result 1 (score: 0.95):
# Citation: [Kubernetes Documentation, Pod Configuration Guide, Section 'Resource Limits']
# Link: https://kubernetes.io/docs/concepts/workloads/pods/#resource-limits
# Content: You can specify resource requests and limits for each container...
```

### API Integration

```python
# Search results now include citation metadata
results = rag_processor.search("kubernetes deployment best practices")

for result in results:
    print(f"Content: {result['content']}")
    
    if 'citation' in result:
        print(f"Citation: {result['citation']['formatted']}")
        if 'link' in result['citation']:
            print(f"Link: {result['citation']['link']}")
```

### Agent Integration

Agents automatically receive citation information and are prompted to include it:

```python
# Agent responses now include citations
response = agent.chat([HumanMessage(content="How do I configure pod resource limits?")])

# Response includes:
# "To configure pod resource limits, you need to specify resources in your container spec [Kubernetes Docs, Pod Configuration]: 'spec.containers[].resources.limits'"
```

## Implementation Details

### Processing Pipeline Integration

The citation system integrates with the existing processing pipeline through the `CitationEnricher`:

```
LoadStage → ValidateStage → SplitStage → EnrichStage → VectorStoreStage
                                             ↓
                                      CitationEnricher
                                      (if enabled)
```

### Document Metadata Enhancement

Each document chunk gets enhanced with citation metadata:

```python
doc.metadata.update({
    'citation_reference': {
        'source_url': 'https://kubernetes.io/docs/concepts/workloads/pods/',
        'resource_type': 'dsl',
        'page_number': None,
        'section_title': 'Pod Configuration',
        'exact_quote': 'You can specify resource requests and limits',
        'chunk_id': 'k8s_docs_abc123',
        'confidence_score': 0.95
    },
    'citation_enabled': True,
    'citation_verbosity': 'detailed',
    'citation_strategy': 'technical_citation'
})
```

### Strategy Pattern

Different citation strategies handle different content types:

- **TechnicalCitationStrategy**: API docs, technical specifications
- **WebCitationStrategy**: Web pages, online documentation  
- **PolicyCitationStrategy**: Policy documents, compliance guides
- **DefaultCitationStrategy**: Fallback for general content

## Examples

### Example 1: Technical Documentation

```python
# Input: Kubernetes documentation about pods
# Strategy: TechnicalCitationStrategy
# Verbosity: FORENSIC

# Output:
"[Kubernetes API v1.28, Pod Specification, Section: Resource Management]: 'spec.containers[].resources.limits.memory'"
```

### Example 2: Web Content

```python
# Input: Blog post about microservices
# Strategy: WebCitationStrategy  
# Verbosity: STANDARD

# Output:
"[martinfowler.com, Microservices Architecture]"
```

### Example 3: Policy Document

```python
# Input: Security policy PDF
# Strategy: PolicyCitationStrategy
# Verbosity: DETAILED

# Output:  
"[Corporate Security Policy v3.2, Section 4.3 'Access Control Requirements', Page 15]"
```

## Troubleshooting

### Common Issues

1. **Citations Not Appearing**
   - Check that `citation.enabled=True` in your config
   - Verify verbosity is not set to `NONE`
   - Ensure resources were processed after enabling citations

2. **Missing Page Numbers**
   - PDF loaders should preserve page metadata
   - Web content won't have page numbers (uses sections instead)

3. **Incorrect Strategy Selection**
   - Check `resource_type` assignment during ingestion
   - Verify strategy mapping in citation config

### Debug Mode

Enable detailed logging to see citation processing:

```python
config.citation.include_confidence = True
config.log_level = "DEBUG"
```

## Future Enhancements

- **Multi-source synthesis tracking**
- **Citation validation and accuracy checking**
- **Interactive citation previews**
- **Alternative source suggestions**
- **Citation export formats (BibTeX, etc.)**

## Migration Guide

### From No Citations to Citations

1. **Enable in Config**:
   ```python
   config.citation.enabled = True
   config.citation.verbosity = CitationVerbosity.STANDARD
   ```

2. **Re-process Resources** (optional for better metadata):
   ```bash
   paas-ai rag sync --force
   ```

3. **Test with Different Verbosity Levels**:
   ```bash
   paas-ai rag search "test query" --config-profile my_citation_profile
   ```

### Gradual Rollout

Start with minimal verbosity and increase based on user feedback:

```python
# Week 1: Minimal citations
citation.verbosity = CitationVerbosity.MINIMAL

# Week 2: Standard citations  
citation.verbosity = CitationVerbosity.STANDARD

# Week 3: Detailed citations for technical content
citation.resource_overrides = {
    ResourceType.DSL: CitationVerbosity.DETAILED
}
```

The citation system provides a robust foundation for source attribution that can grow with your needs while maintaining full backward compatibility. 