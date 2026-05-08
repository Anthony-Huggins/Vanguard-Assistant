"""LLM factory. Returns a LangChain ``BaseChatModel``.

Selecting a provider is a .env change — never edit agent code.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from ..settings import get_settings


def build_chat_model(
    *,
    max_tokens: int | None = None,
    model_override: str | None = None,
) -> BaseChatModel:
    """Construct the chat model for the configured provider.

    ``max_tokens`` lets the coordinator and subagents pick different caps;
    when omitted it defaults to the subagent cap (the looser one).
    ``model_override`` bypasses ``settings.llm_model``; used by the eval
    judge factory so M12 can score with gpt-4.1 regardless of the app default.
    """
    s = get_settings()
    cap = max_tokens if max_tokens is not None else s.llm_max_tokens_subagent

    if s.llm_provider == "azure_openai":
        return _build_azure_openai(cap, deployment_override=model_override)
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_override or s.llm_model,
            temperature=s.llm_temperature,
            max_tokens=cap,
        )
    if s.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic  # type: ignore[import-not-found]

        return ChatAnthropic(
            model=model_override or s.llm_model,
            temperature=s.llm_temperature,
            max_tokens=cap,
        )

    raise ValueError(f"Unsupported llm_provider: {s.llm_provider!r}")


def _build_azure_openai(
    max_tokens: int,
    deployment_override: str | None = None,
) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    s = get_settings()
    deployment = deployment_override or s.azure_openai_deployment
    if not s.azure_openai_endpoint or not deployment:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT must be set "
            "when LLM_PROVIDER=azure_openai."
        )

    kwargs: dict = {
        "azure_endpoint": s.azure_openai_endpoint,
        "api_version": s.azure_openai_api_version,
        "azure_deployment": deployment,
        "temperature": s.llm_temperature,
        "max_tokens": max_tokens,
    }

    if s.azure_openai_api_key:
        kwargs["api_key"] = s.azure_openai_api_key
    else:
        # No API key → mint an Azure AD token via DefaultAzureCredential
        # (managed identity, az login, service principal, …).
        #
        # Why a static token instead of ``azure_ad_token_provider``?
        # ``langchain-openai`` >=1.x and ``openai`` >=2.x had churn around
        # passing the provider callable; certain combinations drop the
        # provider on the floor and the underlying ``OpenAI()`` constructor
        # raises "Missing credentials".  Pre-resolving to a string and
        # passing it as ``azure_ad_token`` works on every version we've
        # tried.  Tokens last ~1 hour; after that the LLM client needs
        # rebuilding (``build_chat_model`` is called per-turn so this
        # auto-refreshes on any new chat request — no process restart
        # needed).
        from azure.identity import DefaultAzureCredential

        token = (
            DefaultAzureCredential()
            .get_token("https://cognitiveservices.azure.com/.default")
            .token
        )
        kwargs["azure_ad_token"] = token

    return AzureChatOpenAI(**kwargs)
