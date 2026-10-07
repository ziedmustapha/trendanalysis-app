import json
import logging

import requests

from config import (
    anthropic_api_key,
    anthropic_configured,
    anthropic_model,
    auth_headers,
    llm_configured,
    llm_provider,
    ollama_base_url,
    ollama_configured,
    ollama_model,
    openai_configured,
    url,
)

_tokenizer = None
_model = None
_embed_cache = {}
_EMBED_CACHE_MAX = 4000


class LLMNotConfigured(RuntimeError):
    pass


OpenAINotConfigured = LLMNotConfigured


def _load_specter():
    """Lazy-load SPECTER so the Flask app can start before the model is downloaded."""
    global _tokenizer, _model
    if _model is None:
        logging.info("Loading allenai/specter embeddings model (first run may download weights)...")
        import torch  # noqa: F401  — imported for side-effect / CUDA setup
        from transformers import AutoModel, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        _model = AutoModel.from_pretrained("allenai/specter")
        logging.info("SPECTER model ready.")
    return _tokenizer, _model


def embed_text(text):
    """
    Embed the given text using a pre-trained model.
    @param text - The input text to be embedded
    @return The embedded representation of the input text
    """
    import torch

    if text is None:
        text = "..."

    cached = _embed_cache.get(text)
    if cached is not None:
        return cached

    tokenizer, model = _load_specter()
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

    if len(_embed_cache) < _EMBED_CACHE_MAX:
        _embed_cache[text] = embedding
    return embedding


def _require_llm():
    provider = llm_provider()
    if provider == "ollama" and not ollama_configured():
        raise LLMNotConfigured(
            "Ollama is not running. Start it with `ollama serve`, then retry Analyze / Generate Trends."
        )
    if not llm_configured():
        raise LLMNotConfigured(
            "No local LLM running. Start Ollama (`ollama serve`) or add ANTHROPIC_API_KEY to .env."
        )


def _summarize_cluster(cluster, max_related=12, abstract_chars=400):
    """Shrink cluster payloads so naming/trends stay cheap and within context."""
    if not isinstance(cluster, (list, tuple)) or len(cluster) < 3:
        return cluster
    query_titles = cluster[0]
    related = cluster[2] or []
    papers = []
    for paper in related[:max_related]:
        title = paper[0] if paper else ""
        abstract = paper[1] if len(paper) > 1 else ""
        score = paper[2] if len(paper) > 2 else None
        papers.append({
            "title": title,
            "abstract": (abstract or "")[:abstract_chars],
            "similarity": score,
        })
    return {
        "new_paper_titles": query_titles,
        "related_older_papers": papers,
    }


def _anthropic_chat(system, user, max_tokens):
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": anthropic_model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=180,
    )
    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"Anthropic API error {response.status_code}: {detail}")
    data = response.json()
    parts = data.get("content") or []
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
    if not text:
        raise RuntimeError(f"Anthropic API returned no text: {data}")
    return text.strip()


def _openai_chat(messages, timeout=120):
    if not openai_configured() or not url:
        raise LLMNotConfigured("OpenAI gateway is not configured.")
    req = requests.post(url, headers=auth_headers, json={"messages": messages}, verify=False, timeout=timeout)
    req.raise_for_status()
    res = req.json()
    return res["choices"][0]["message"]["content"].strip()


def _ollama_chat(system, user, max_tokens):
    response = requests.post(
        f"{ollama_base_url}/api/chat",
        json={
            "model": ollama_model,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text[:500]}")
    data = response.json()
    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        raise RuntimeError(f"Ollama returned no text: {data}")
    return text


def _complete(system, user, max_tokens=400):
    _require_llm()
    provider = llm_provider()
    if provider == "ollama":
        return _ollama_chat(system, user, max_tokens=max_tokens)
    if provider == "anthropic":
        return _anthropic_chat(system, user, max_tokens=max_tokens)
    return _openai_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        timeout=180,
    )


def generate_cluster_name(cluster):
    """Generate a short cluster name from new papers and their similar older papers."""
    payload = json.dumps(_summarize_cluster(cluster), ensure_ascii=False)
    name = _complete(
        "You name academic-paper clusters. Reply with only the name, no quotes or extra text.",
        "Give a concise name (at most 8 words) for this cluster of new papers and their most similar older papers:\n"
        + payload,
        max_tokens=80,
    )
    return f'"{name.strip().strip(chr(34))}"'


def identify_trends_in_cluster(cluster):
    """Identify trends in a cluster. Return HTML-ish bullets the frontend already parses."""
    payload = json.dumps(_summarize_cluster(cluster), ensure_ascii=False)
    return _complete(
        "Analyze new academic papers against similar older papers. Identify trends, shifts in research focus, and thematic changes.",
        """Here is the data of new academic papers and their most similar but older papers. Analyze and extract significant trends.
The response must be of this form:
- <b>"Category of trend 1"</b>: ... / <b>Importance degree: "score"</b> <br>
- <b>"Category of trend 2"</b>: ... / <b>Importance degree: "score"</b> <br>
Do not add anything more to your response.
The importance degree score must be Low, Medium, or High.
Sort trends from highest to lowest importance.
Each item should be a short paragraph.
Categorize each trend as one of:
- An opposition
- A shift of focus on a specific domain
- Emerging topic
- An increase of papers talking about the same topic
- Technological advances
- Regulatory Changes
- Geographical shifts
- Methodological innovations
- Public Health Concerns
- Risk Assessment and Management
- Consumer behavior and consumption
- Environmental impact
- Economic
- Cultural and Sociopolitical Factors

Data:
""" + payload,
        max_tokens=1200,
    )
