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


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Copy .env.example to .env and fill in your secrets."
        )
    return value


client_id = _require_env("CLIENT_ID")
client_secret = _require_env("CLIENT_SECRET")
deploymentName = os.getenv("DEPLOYMENT_NAME", "ChatGPT")
action = "chat"
action_extension = "completions"
api_version = os.getenv("API_VERSION", "2023-07-01-preview")
openai_api_base_url = _require_env("OPENAI_API_BASE_URL").rstrip("/")


def build_openai_url(deployment: str, action_name: str, action_ext: Optional[str] = None) -> str:
    """Build a Nestlé OpenAI gateway URL from env-configured base settings."""
    path = f"{openai_api_base_url}/deployments/{deployment}/{action_name}"
    if action_ext:
        path = f"{path}/{action_ext}"
    return f"{path}?api-version={api_version}"


url = build_openai_url(deploymentName, action, action_extension)

auth_headers = {
    "client_id": client_id,
    "client_secret": client_secret,
}
