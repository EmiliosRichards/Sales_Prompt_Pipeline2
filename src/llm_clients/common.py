from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UsageMetadata:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": int(self.prompt_token_count or 0),
            "completion_tokens": int(self.candidates_token_count or 0),
            "total_tokens": int(self.total_token_count or 0),
        }


@dataclass
class CompatLLMResponse:
    text: str
    usage_metadata: UsageMetadata = field(default_factory=UsageMetadata)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    parsed: Optional[Any] = None
    prompt_feedback: Optional[Any] = None
    status: str = "success"
    refusal: Optional[str] = None
    provider_error: Optional[str] = None
    retry_count: int = 0
    model_name: Optional[str] = None


@dataclass
class StructuredLLMResult:
    parsed_output: Optional[Any]
    raw_text: str
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    })
    status: str = "success"
    refusal: Optional[str] = None
    provider_error: Optional[str] = None
    retry_count: int = 0
    model_name: Optional[str] = None
