from loguru import logger
from dotenv import load_dotenv
import uuid
import litellm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
import warnings
load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Anima:
    """
    Core class for the animax package. Serves as the main entry point for anima-related operations.

    Attributes:
        name: The name of the anima instance.
        config: Optional configuration dictionary for customization.
    """
    def __init__(
        self,
        name: str = None,
        backend: str = "litellm",
        model_endpoint_style: str = "openai",
        model_name: str = "gpt-4o",
        api_key: str = None,
        api_base: str = None,
        system_prompt: str = None,
        previous_messages: list[dict[str, str]] = None,
    ):
        self._id: str = str(uuid.uuid4())
        logger.debug(f"Generated unique Anima ID: {self._id}")
        self.name: str = name if name is not None else "anima-" + self._id[:8]
        logger.info(f"Initializing Anima instance with name: {self.name}")
        self.backend: str = backend
        logger.debug(f"Backend set to: {self.backend}")
        self.model_endpoint_style: str = model_endpoint_style
        logger.debug(f"Model endpoint style set to: {self.model_endpoint_style}")
        tokens = model_name.split("/", 1)
        if len(tokens) == 2 and tokens[0] == self.model_endpoint_style:
            self.model_name = tokens[1]
            logger.debug(f"Model name reformatted from '{model_name}' to '{self.model_name}'.")
        else:
            self.model_name = model_name
            logger.debug(f"Model name set to: {self.model_name}")
        self.api_key: str = api_key
        if self.api_key is not None:
            logger.debug("API key provided.")
        else:
            logger.warning("No API key provided.")
        self.api_base: str = api_base
        if self.api_base is not None:
            logger.debug(f"API base set to: {self.api_base}")
        self.system_prompt: str = system_prompt
        if self.system_prompt is not None:
            logger.debug(f"System prompt set to: {self.system_prompt}")
        self.previous_messages: list[dict[str, str]] = previous_messages if previous_messages is not None else []

    def _prepare_messages(self, prompt: str | None, system_prompt: str | None) -> list[dict[str, str]]:
        """Prepare messages list for inference from prompt and system_prompt."""
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        elif self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        return messages

    def _prepare_chat_messages(self, prompt: str) -> list[dict[str, str]]:
        """Compose the full chat message history including system prompt and conversation history."""
        system_messages: list[dict[str, str]] = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt is not None else [])
        previous_messages = self.previous_messages if self.previous_messages is not None else []
        return system_messages + previous_messages + [{"role": "user", "content": prompt}]

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type(Exception), reraise=True)
    def _call_litellm(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        n: int,
        stop: list[str] | None,
        timeout: float | None
    ) -> dict:
        logger.info(f"Calling litellm.completion (attempt with first message: {messages[0] if messages else 'EMPTY'})")
        try:
            response: dict = litellm.completion(
                model=f"{self.model_endpoint_style}/{self.model_name}",
                api_key=self.api_key,
                api_base=self.api_base,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                stop=stop,
                timeout=timeout
            )
            logger.debug(f"litellm.completion response: {response}")
            return response
        except Exception as exc:
            logger.error(f"litellm.completion failed: {exc}")
            raise

    def _do_inference(
        self,
        prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        system_prompt: str | None = None,
        n: int = 1,
        stop: list[str] | None = None,
        timeout: float | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> dict:
        if messages is None:
            logger.debug(f"Preparing to perform inference with prompt: {prompt}")
            messages = self._prepare_messages(prompt, system_prompt)
        else:
            logger.debug(f"Using provided message history for inference: {messages}")
        try:
            response: dict = self._call_litellm(messages, max_tokens, temperature, n, stop, timeout)
            logger.debug("Inference successful.")
            return response
        except RetryError as re:
            logger.critical(f"All retry attempts for inference failed: {re}")
            raise
        except Exception as exc:
            logger.critical(f"Inference failed with unexpected error: {exc}")
            raise

    def _validate_litellm_backend(self) -> bool:
        """
        Validate the litellm backend by making a minimal completion call.
        Logs the attempt and result, and errors if any occur.

        Returns:
            True if the backend is valid, False otherwise.
        """
        logger.info("Validating litellm backend connectivity and credentials.")
        try:
            response: dict = self._do_inference(prompt="Hi", max_tokens=1)
            valid: bool = bool(response.get("choices") and response["choices"][0].get("message", {}).get("content"))
            logger.debug(f"litellm backend validation response: {response}")
            logger.info(f"litellm backend validation result: {valid}")
            return valid
        except Exception as exc:
            logger.error(f"Error validating litellm backend: {exc}")
            return False

    def validate_backend(self):
        """
        Validate the configured backend. Logs the backend being validated and errors for unsupported backends.
        """
        logger.info(f"Validating backend: {self.backend}")
        if self.backend == "litellm":
            return self._validate_litellm_backend()
        else:
            logger.error(f"Backend {self.backend} is not supported")
            raise ValueError(f"Backend {self.backend} is not supported")

    def ask(self, prompt: str, max_tokens: int = 4096, temperature: float = 1.0) -> str:
        """
        Ask the anima a question.

        Args:
            prompt: The user prompt to send to the model.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The model's response as a string.
        """
        logger.debug(f"Asking anima: {prompt}")
        try:
            response: dict = self._do_inference(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            answer: str = response["choices"][0]["message"]["content"]
            logger.debug(f"Anima response: {answer}")
            return answer
        except Exception as exc:
            logger.error(f"Failed to get response from anima: {exc}")
            return "[Error: Failed to get response from anima]"

    def chat(self, prompt: str, max_tokens: int = 4096, temperature: float = 1.0) -> str:
        """
        Chat with the anima. Maintains conversation history by storing previous messages and appending new exchanges.

        Args:
            prompt: The user prompt to send to the model.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The model's response as a string.
        """
        logger.debug(f"Chatting with anima: {prompt}")
        if self.previous_messages is None:
            self.previous_messages = []
        messages: list[dict[str, str]] = self._prepare_chat_messages(prompt)
        try:
            response: dict = self._do_inference(max_tokens=max_tokens, temperature=temperature, messages=messages)
            answer: str = response["choices"][0]["message"]["content"]
            # Update conversation history without duplicating the system prompt
            self.previous_messages.extend([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ])
            logger.debug(f"Updated conversation history: {self.previous_messages}")
            return answer
        except Exception as exc:
            logger.error(f"Failed to get response from anima during chat: {exc}")
            return "[Error: Failed to get response from anima]"

if __name__ == "__main__":
    # This block is for demonstration purposes when anima.py is run directly
    anima = Anima()
    result = anima.validate_backend()
    logger.info(f"Backend validation result: {result}")
    if result:
        response = anima.ask("Hello, how are you?")
        logger.info(f"Anima response: {response}")
