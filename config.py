import os
from pathlib import Path
from typing import Optional


def _load_dotenv_file(env_path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no overwrite)."""
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except ImportError:
        pass

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


_load_dotenv_file(Path(__file__).resolve().parent / ".env")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


client_id = _env("CLIENT_ID")
client_secret = _env("CLIENT_SECRET")
deploymentName = _env("DEPLOYMENT_NAME", "ChatGPT")
action = "chat"
action_extension = "completions"
api_version = _env("API_VERSION", "2023-07-01-preview")
openai_api_base_url = _env("OPENAI_API_BASE_URL").rstrip("/")

anthropic_api_key = _env("ANTHROPIC_API_KEY")
anthropic_model = _env("ANTHROPIC_MODEL", "claude-haiku-4-5")
ollama_base_url = _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
ollama_model = _env("OLLAMA_MODEL", "qwen2.5:3b")
llm_provider_override = _env("LLM_PROVIDER").lower()


def openai_configured() -> bool:
    placeholder_values = {
        "",
        "your_client_id_here",
        "your_client_secret_here",
        "https://your-gateway.example.com/api/your-openai-api/1/openai",
    }
    return (
        client_id not in placeholder_values
        and client_secret not in placeholder_values
        and openai_api_base_url not in placeholder_values
    )


def anthropic_configured() -> bool:
    key = anthropic_api_key
    return bool(key) and not key.startswith("your_") and key != "sk-ant-your-key-here"


def ollama_configured() -> bool:
    try:
        from urllib.request import urlopen

        with urlopen(f"{ollama_base_url}/api/tags", timeout=0.6) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception:
        return False


def llm_provider() -> str:
    if llm_provider_override in {"ollama", "anthropic", "openai"}:
        return llm_provider_override
    if ollama_configured():
        return "ollama"
    if anthropic_configured():
        return "anthropic"
    if openai_configured():
        return "openai"
    return ""


def llm_configured() -> bool:
    if llm_provider_override == "ollama":
        return True
    return bool(llm_provider())


def build_openai_url(deployment: str, action_name: str, action_ext: Optional[str] = None) -> str:
    """Build a Nestlé OpenAI gateway URL from env-configured base settings."""
    if not openai_api_base_url:
        return ""
    path = f"{openai_api_base_url}/deployments/{deployment}/{action_name}"
    if action_ext:
        path = f"{path}/{action_ext}"
    return f"{path}?api-version={api_version}"


url = build_openai_url(deploymentName, action, action_extension)

auth_headers = {
    "client_id": client_id,
    "client_secret": client_secret,
}
