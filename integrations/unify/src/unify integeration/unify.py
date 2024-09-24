from typing import Any, Callable, ClassVar, Dict, List, Optional
from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret
from unify import Unify

logger = logging.getLogger(__name__)

@component
class UnifyGenerator:
    """
    Integrates Unify LLM API for text generation using models like 'mistral-7b-instruct-v0.2@fireworks-ai'.
    """

    ALLOWED_PARAMS: ClassVar[List[str]] = ["max_tokens", "temperature"]

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("UNIFY_API_KEY"),
        model: str = "mistral-7b-instruct-v0.2@fireworks-ai",
        streaming_callback: Optional[Callable[[str], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the UnifyGenerator component.

        :param api_key: Unify API key.
        :param model: Unify model name.
        :param streaming_callback: Optional callback for streamed responses.
        :param generation_kwargs: Additional parameters for generation.
        """
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback
        self.client = Unify(model=self.model, api_key=self.api_key.resolve_value())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize component to a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifyGenerator":
        """
        Deserialize from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)
    

    @component.output_types(replies=List[str], meta=Dict[str, Any])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Generate responses using Unify API.

        :param prompt: Input prompt for generation.
        :param generation_kwargs: Additional parameters for generation.
        :returns: Dictionary with replies and metadata.
        """
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        filtered_kwargs = {k: v for k, v in generation_kwargs.items() if k in self.ALLOWED_PARAMS}

        response = self.client.generate(prompt=prompt, **filtered_kwargs)
        return {"replies": [response["text"]], "meta": response.get("meta", {})}
