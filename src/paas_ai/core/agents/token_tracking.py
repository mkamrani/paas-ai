"""
Token tracking system for multi-agent operations.

Provides token usage tracking with pluggable callback system for external integrations.
"""

import time
import uuid
import json
import asyncio
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Protocol
from pathlib import Path

from paas_ai.utils.logging import get_logger

logger = get_logger("paas_ai.agents.token_tracking")


@dataclass
class TokenUsage:
    """Token usage information for a single operation."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    agent: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    processing_time: Optional[float] = None


class TokenUsageCallback(Protocol):
    """Protocol for token usage callbacks."""
    
    def on_token_usage(self, usage: TokenUsage) -> None:
        """Called when tokens are consumed."""
        ...
    
    def on_session_end(self, summary: Dict[str, Any]) -> None:
        """Called when a session ends (optional)."""
        ...


class SessionTokenTracker:
    """
    Tracks token usage for a session with optional callback integration.
    
    Accumulates token usage across multiple agent calls and provides
    session summaries while triggering callbacks for external systems.
    """
    
    def __init__(
        self, 
        enabled: bool = False,
        callback: Optional[TokenUsageCallback] = None,
        verbose: bool = False,
        session_id: Optional[str] = None
    ):
        """
        Initialize session token tracker.
        
        Args:
            enabled: Whether token tracking is enabled
            callback: Optional callback for token usage events
            verbose: Whether to log token usage verbosely
            session_id: Unique session identifier
        """
        self.enabled = enabled
        self.callback = callback
        self.verbose = verbose
        self.session_id = session_id or str(uuid.uuid4())
        self.usage_history: List[TokenUsage] = []
        self.session_start_time = time.time()
    
    def track(
        self, 
        usage_data: Dict[str, int], 
        model: str, 
        agent: str, 
        request_id: Optional[str] = None,
        processing_time: Optional[float] = None,
        **context
    ) -> None:
        """
        Track token usage for an operation.
        
        Args:
            usage_data: Dictionary with input_tokens, output_tokens, total_tokens
            model: Model name that generated the tokens
            agent: Agent that made the request
            request_id: Optional request identifier
            processing_time: Time taken to process the request
            **context: Additional context for the usage record
        """
        if not self.enabled:
            return
        
        token_usage = TokenUsage(
            input_tokens=usage_data.get('input_tokens', 0),
            output_tokens=usage_data.get('output_tokens', 0),
            total_tokens=usage_data.get('total_tokens', 0),
            model=model,
            agent=agent,
            session_id=self.session_id,
            request_id=request_id,
            processing_time=processing_time
        )
        
        self.usage_history.append(token_usage)
        
        # Verbose logging
        if self.verbose:
            logger.info(
                f"🪙 Token usage - {agent}: {token_usage.total_tokens} tokens "
                f"({model}) [req: {request_id or 'unknown'}]"
            )
        
        # Trigger callback
        if self.callback:
            try:
                self.callback.on_token_usage(token_usage)
            except Exception as e:
                logger.warning(f"Token callback failed: {e}")
    
    def track_direct(self, token_usage: TokenUsage) -> None:
        """Track a pre-built TokenUsage object directly."""
        if not self.enabled:
            return
        
        self.usage_history.append(token_usage)
        
        if self.verbose:
            logger.info(
                f"🪙 Token usage - {token_usage.agent}: {token_usage.total_tokens} tokens "
                f"({token_usage.model})"
            )
        
        if self.callback:
            try:
                self.callback.on_token_usage(token_usage)
            except Exception as e:
                logger.warning(f"Token callback failed: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session token usage summary."""
        if not self.enabled:
            return {
                "session_id": self.session_id,
                "total_tokens": 0,
                "total_requests": 0,
                "enabled": self.enabled,
                "session_duration": time.time() - self.session_start_time
            }
        
        if not self.usage_history:
            return {
                "session_id": self.session_id,
                "total_tokens": 0,
                "total_requests": 0,
                "enabled": self.enabled,
                "session_duration": time.time() - self.session_start_time,
                "agent_breakdown": {},
                "model_breakdown": {},
                "agents_used": [],
                "models_used": []
            }
        
        # Calculate totals
        total_input = sum(usage.input_tokens for usage in self.usage_history)
        total_output = sum(usage.output_tokens for usage in self.usage_history)
        total_tokens = sum(usage.total_tokens for usage in self.usage_history)
        
        # Agent breakdown
        agent_breakdown = {}
        for usage in self.usage_history:
            if usage.agent not in agent_breakdown:
                agent_breakdown[usage.agent] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0
                }
            
            agent_breakdown[usage.agent]["input_tokens"] += usage.input_tokens
            agent_breakdown[usage.agent]["output_tokens"] += usage.output_tokens
            agent_breakdown[usage.agent]["total_tokens"] += usage.total_tokens
            agent_breakdown[usage.agent]["requests"] += 1
        
        # Model breakdown
        model_breakdown = {}
        for usage in self.usage_history:
            if usage.model not in model_breakdown:
                model_breakdown[usage.model] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "requests": 0
                }
            
            model_breakdown[usage.model]["input_tokens"] += usage.input_tokens
            model_breakdown[usage.model]["output_tokens"] += usage.output_tokens
            model_breakdown[usage.model]["total_tokens"] += usage.total_tokens
            model_breakdown[usage.model]["requests"] += 1
        
        # Last request info
        last_usage = self.usage_history[-1] if self.usage_history else None
        
        return {
            "session_id": self.session_id,
            "session_duration": time.time() - self.session_start_time,
            "total_requests": len(self.usage_history),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "agent_breakdown": agent_breakdown,
            "model_breakdown": model_breakdown,
            "last_tokens": last_usage.total_tokens if last_usage else 0,
            "agents_used": list(agent_breakdown.keys()),
            "models_used": list(model_breakdown.keys())
        }
    
    def get_last_request_summary(self) -> Dict[str, Any]:
        """Get summary for the most recent request."""
        if not self.enabled or not self.usage_history:
            return {"total_tokens": 0}
        
        last_usage = self.usage_history[-1]
        return {
            "input_tokens": last_usage.input_tokens,
            "output_tokens": last_usage.output_tokens,
            "total_tokens": last_usage.total_tokens,
            "model": last_usage.model,
            "agent": last_usage.agent,
            "processing_time": last_usage.processing_time
        }
    
    def clear(self) -> None:
        """Clear session history."""
        if self.callback and hasattr(self.callback, 'on_session_end'):
            try:
                summary = self.get_session_summary()
                self.callback.on_session_end(summary)
            except Exception as e:
                logger.warning(f"Session end callback failed: {e}")
        
        self.usage_history.clear()
        self.session_start_time = time.time()
    
    def end_session(self) -> Dict[str, Any]:
        """End the session and trigger final callback."""
        summary = self.get_session_summary()
        
        if self.callback and hasattr(self.callback, 'on_session_end'):
            try:
                self.callback.on_session_end(summary)
            except Exception as e:
                logger.warning(f"Session end callback failed: {e}")
        
        return summary


class TokenCallbackFactory:
    """Factory for creating token usage callbacks."""
    
    _callbacks: Dict[str, Type[TokenUsageCallback]] = {}
    
    @classmethod
    def register(cls, name: str, callback_class: Type[TokenUsageCallback]) -> None:
        """Register a callback implementation."""
        cls._callbacks[name] = callback_class
        logger.debug(f"Registered token callback: {name}")
    
    @classmethod
    def get_callback_class(cls, name: str) -> Optional[Type[TokenUsageCallback]]:
        """Get a callback class by name."""
        return cls._callbacks.get(name)
    
    @classmethod
    def create_callback(cls, name: str, **kwargs) -> Optional[TokenUsageCallback]:
        """Create a callback instance by name."""
        callback_class = cls.get_callback_class(name)
        if callback_class:
            try:
                return callback_class(**kwargs)
            except Exception as e:
                logger.error(f"Failed to create callback '{name}': {e}")
                return None
        
        logger.warning(f"Unknown callback: {name}")
        return None
    
    @classmethod
    def list_callbacks(cls) -> List[str]:
        """List all registered callback names."""
        return list(cls._callbacks.keys())


def register_callback(name: str):
    """Decorator to register a callback class."""
    def decorator(callback_class: Type[TokenUsageCallback]):
        TokenCallbackFactory.register(name, callback_class)
        return callback_class
    return decorator


# Built-in callback implementations

@register_callback("console")
class ConsoleTokenCallback:
    """Simple console logging callback."""
    
    def __init__(self, verbose: bool = False, show_details: bool = False):
        """
        Initialize console callback.
        
        Args:
            verbose: Whether to show detailed token info
            show_details: Whether to show processing time and model details
        """
        self.verbose = verbose
        self.show_details = show_details
    
    def on_token_usage(self, usage: TokenUsage) -> None:
        """Log token usage to console."""
        if not self.verbose:
            return
        
        if self.show_details:
            processing_info = f" ({usage.processing_time:.2f}s)" if usage.processing_time else ""
            print(f"🪙 {usage.agent}: {usage.total_tokens} tokens ({usage.model}){processing_info}")
        else:
            print(f"🪙 {usage.agent}: {usage.total_tokens} tokens")
    
    def on_session_end(self, summary: Dict[str, Any]) -> None:
        """Log session summary to console."""
        if self.verbose:
            total_tokens = summary.get('total_tokens', 0)
            agents = summary.get('agents_used', [])
            duration = summary.get('session_duration', 0)
            
            print(f"📊 Session complete: {total_tokens} tokens across {len(agents)} agents ({duration:.1f}s)")


@register_callback("json_file")
class JsonFileTokenCallback:
    """Log token usage to JSONL file for analysis."""
    
    def __init__(self, file_path: str = "token_usage.jsonl", include_session_summary: bool = True):
        """
        Initialize file callback.
        
        Args:
            file_path: Path to the JSONL file
            include_session_summary: Whether to log session summaries
        """
        self.file_path = Path(file_path)
        self.include_session_summary = include_session_summary
        
        # Ensure directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    def on_token_usage(self, usage: TokenUsage) -> None:
        """Log token usage to file."""
        try:
            usage_dict = asdict(usage)
            usage_dict['timestamp'] = usage.timestamp.isoformat()
            usage_dict['event_type'] = 'token_usage'
            
            with open(self.file_path, 'a') as f:
                json.dump(usage_dict, f, default=str)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write token usage to file: {e}")
    
    def on_session_end(self, summary: Dict[str, Any]) -> None:
        """Log session summary to file."""
        if not self.include_session_summary:
            return
        
        try:
            summary_record = {
                "event_type": "session_end",
                "timestamp": datetime.utcnow().isoformat(),
                **summary
            }
            
            with open(self.file_path, 'a') as f:
                json.dump(summary_record, f, default=str)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write session summary to file: {e}")


@register_callback("webhook")
class WebhookTokenCallback:
    """Send token usage to external webhook."""
    
    def __init__(
        self, 
        webhook_url: str, 
        api_key: Optional[str] = None,
        timeout: int = 10,
        retry_attempts: int = 1,
        send_session_summary: bool = True
    ):
        """
        Initialize webhook callback.
        
        Args:
            webhook_url: URL to send token data to
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
            send_session_summary: Whether to send session summaries
        """
        self.webhook_url = webhook_url
        self.api_key = api_key
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.send_session_summary = send_session_summary
    
    def on_token_usage(self, usage: TokenUsage) -> None:
        """Send token usage to webhook (async)."""
        usage_dict = asdict(usage)
        usage_dict['timestamp'] = usage.timestamp.isoformat()
        usage_dict['event_type'] = 'token_usage'
        
        # Send asynchronously to avoid blocking
        asyncio.create_task(self._send_webhook(usage_dict))
    
    def on_session_end(self, summary: Dict[str, Any]) -> None:
        """Send session summary to webhook."""
        if not self.send_session_summary:
            return
        
        summary_record = {
            "event_type": "session_end",
            "timestamp": datetime.utcnow().isoformat(),
            **summary
        }
        
        # Send asynchronously
        asyncio.create_task(self._send_webhook(summary_record))
    
    async def _send_webhook(self, data: Dict[str, Any]) -> None:
        """Send data to webhook with retries."""
        try:
            import aiohttp
            
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            for attempt in range(self.retry_attempts):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            self.webhook_url, 
                            json=data, 
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=self.timeout)
                        ) as response:
                            if response.status < 400:
                                logger.debug(f"Token data sent to webhook: {data.get('event_type')}")
                                return
                            else:
                                logger.warning(f"Webhook returned status {response.status}")
                
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        logger.error(f"Failed to send token data to webhook after {self.retry_attempts} attempts: {e}")
                    else:
                        logger.debug(f"Webhook attempt {attempt + 1} failed: {e}")
                        await asyncio.sleep(1)  # Brief delay before retry
        
        except ImportError:
            logger.error("aiohttp not available for webhook callback")
        except Exception as e:
            logger.error(f"Unexpected error in webhook callback: {e}")


# Initialize built-in callbacks
def _register_built_in_callbacks():
    """Register all built-in callbacks."""
    # They're already registered via decorators above
    logger.info(f"Token callback system initialized with {len(TokenCallbackFactory.list_callbacks())} built-in callbacks")


# Register on module import
_register_built_in_callbacks() 