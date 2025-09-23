"""
Document validation stage for processing pipeline.
"""

from typing import List
from langchain_core.documents import Document

from ..base import ProcessingStage, ProcessingContext
from ...content_validators import ContentValidatorFactory
from ....config import ContentValidatorConfig, ContentValidatorType


class ValidateStage(ProcessingStage):
    """Stage for validating documents using content validator strategies."""
    
    def __init__(self, 
                 name: str = "validate",
                 validator_config: ContentValidatorConfig = None):
        super().__init__(name)
        
        # Use default content validator config if none provided
        if validator_config is None:
            validator_config = ContentValidatorConfig(
                type=ContentValidatorType.CONTENT,
                min_content_length=10,
                max_content_length=1000000,
                skip_empty=True
            )
        
        self.validator_config = validator_config
        self.content_validator = ContentValidatorFactory.create_content_validator(validator_config)
    
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """Validate documents using the content validator strategy."""
        documents = context.documents
        
        if not documents:
            # Set metadata even when no documents
            context.metadata.update({
                'validated_count': 0,
                'rejected_count': 0,
                'validation_rules': {
                    'type': self.validator_config.type,
                    'min_length': self.validator_config.min_content_length,
                    'max_length': self.validator_config.max_content_length,
                    'skip_empty': self.validator_config.skip_empty
                }
            })
            return context
        
        initial_count = len(documents)
        valid_documents = self.content_validator.validate_documents(documents, self.validator_config)
        rejected_count = initial_count - len(valid_documents)
        
        # Update context
        context.documents = valid_documents
        context.metadata.update({
            'validated_count': len(valid_documents),
            'rejected_count': rejected_count,
            'validation_rules': {
                'type': self.validator_config.type,
                'min_length': self.validator_config.min_content_length,
                'max_length': self.validator_config.max_content_length,
                'skip_empty': self.validator_config.skip_empty
            }
        })
        
        return context 