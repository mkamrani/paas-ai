"""
Factory for creating content validators based on configuration.
"""

from .base import ContentValidatorStrategy
from .content import ContentValidator
from ...config import ContentValidatorConfig, ContentValidatorType


class ContentValidatorFactory:
    """Factory for creating content validators based on configuration."""
    
    @staticmethod
    def create_content_validator(config: ContentValidatorConfig) -> ContentValidatorStrategy:
        """Create a content validator based on configuration."""
        validator_type = config.type
        
        if validator_type == ContentValidatorType.CONTENT:
            validator = ContentValidator()
        else:
            raise ValueError(f"Unsupported content validator type: {validator_type}")
        
        # Validate configuration
        validator.validate_config(config)
        
        return validator 