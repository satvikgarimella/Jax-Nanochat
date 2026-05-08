from __future__ import annotations

import asyncio
import re
from typing import Any

DEFAULT_TEMPERATURE = 0.6
FACTUAL_TEMPERATURE_CAP = 0.55
CREATIVE_TEMPERATURE_CAP = 0.85
FACTUAL_MAX_TOKENS = 120
CREATIVE_MAX_TOKENS = 160
MAX_HISTORY_TURNS = 8
WEB_CONTEXT_CHARS = 320

FACTUAL_PREFIXES = (
    "what is",
    "what are",
    "what was",
    "what were",
    "who is",
    "who was",
    "who are",
    "when did",
    "when was",
    "when is",
    "where is",
    "where was",
    "where are",
    "how does",
    "how do",
    "how did",
    "explain ",
    "define ",
    "tell me about",
    "whats a ",
    "whats the ",
    "what's a ",
    "what's the ",
    "can you tell me",
    "can you explain",
    "can you describe",
    "could you explain",
    "could you tell me",
    "i want to know",
    "i'd like to know",
    "tell me",
    "describe ",
)

FACTUAL_KEYWORDS = (
    "framework",
    "algorithm",
    "definition",
    "difference",
    "history",
    "meaning",
    "example",
    "overview",
    "summary",
    "how to",
    "why is",
    "what does",
)

CREATIVE_PREFIXES = (
    "write ",
    "create ",
    "generate ",
    "make ",
    "compose ",
    "give me a ",
    "tell me a ",
    "come up with",
)

MATH_PREFIXES = (
    "what is",
    "what's",
    "whats",
    "calculate",
    "compute",
    "solve",
    "evaluate",
)

EXTRACTIVE_FACTUAL_PREFIXES = (
    "what is ",
    "what are ",
    "what's ",
    "whats ",
    "define ",
)

NON_EXTRACTIVE_MARKERS = (
    ",",
    " and ",
    " where ",
    " when ",
    " why ",
    " how ",
    " example",
    " examples",
    " compare ",
    " difference ",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "could",
    "describe",
    "define",
    "does",
    "did",
    "explain",
    "for",
    "how",
    "i",
    "id",
    "is",
    "it",
    "know",
    "like",
    "me",
    "of",
    "or",
    "please",
    "tell",
    "the",
    "this",
    "to",
    "want",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "you",
}

CANONICAL_DEFINITIONS = {
    "artificial intelligence": "Artificial intelligence is the field of building systems that can perform tasks that normally require human intelligence.",
    "machine learning": "Machine learning is a branch of AI where algorithms learn patterns from data so they can make predictions or decisions without being explicitly programmed for every case.",
    "deep learning": "Deep learning is a type of machine learning that uses multi-layer neural networks to learn complex patterns from large amounts of data.",
    "neural network": "A neural network is a model made of connected layers of simple units that learn patterns from data.",
    "transformer": "A transformer is a neural network architecture that uses attention to model relationships between tokens in parallel.",
    "large language model": "A large language model is a neural network trained on a huge amount of text so it can understand and generate language.",
    "tokenizer": "A tokenizer is the component that splits text into tokens that a model can process.",
    "embedding": "An embedding is a numeric vector that represents the meaning or context of a token, word, or item.",
    "tensorflow": "TensorFlow is an open-source machine learning framework from Google for building and training neural networks.",
    "jax": "JAX is a Python library for high-performance numerical computing with automatic differentiation and XLA compilation.",
    "reinforcement learning": "Reinforcement learning is a type of machine learning where an agent learns by taking actions and receiving rewards or penalties.",
    "supervised learning": "Supervised learning is a type of machine learning where a model learns from labeled examples.",
    "unsupervised learning": "Unsupervised learning is a type of machine learning where a model finds patterns or structure in unlabeled data.",
}

SUBJECT_ALIASES = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "neural networks": "neural network",
    "transformers": "transformer",
    "large language models": "large language model",
    "language model": "large language model",
    "language models": "large language model",
    "llm": "large language model",
    "llms": "large language model",
    "tokenizers": "tokenizer",
    "embeddings": "embedding",
    "tensor flow": "tensorflow",
}


def try_solve_math(message: str) -> str | None:
    msg = message.lower().strip().rstrip("?").strip()
    for prefix in MATH_PREFIXES:
        if msg.startswith(prefix):
            msg = msg[len(prefix):].strip()
            break

    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\%\^]+", msg or ""):
        return None
    if not any(ch.isdigit() for ch in msg):
        return None

    try:
        result = eval(msg.replace("^", "**"), {"__builtins__": {}}, {})  # noqa: S307
    except Exception:
        return None

    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def is_creative_request(message: str) -> bool:
    msg = message.lower().strip()
    return any(msg.startswith(prefix) for prefix in CREATIVE_PREFIXES)


def is_factual_question(message: str) -> bool:
    msg = message.lower().strip()
    if is_creative_request(msg):
        return False
    if any(msg.startswith(prefix) for prefix in FACTUAL_PREFIXES):
        return True
    return any(keyword in msg for keyword in FACTUAL_KEYWORDS)


def normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for turn in (history or [])[-MAX_HISTORY_TURNS:]:
        role = "assistant" if turn.get("role") == "assistant" else "user"
        content = str(turn.get("content", "")).strip()
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def should_use_extractive_answer(message: str) -> bool:
    msg = message.lower().strip()
    if not any(msg.startswith(prefix) for prefix in EXTRACTIVE_FACTUAL_PREFIXES):
        return False
    return not any(marker in msg for marker in NON_EXTRACTIVE_MARKERS)


def extract_question_terms(message: str) -> list[str]:
    terms = []
    for token in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", message.lower()):
        if token not in STOPWORDS:
            terms.append(token)
    return terms


def extract_question_subject(message: str) -> str:
    msg = message.lower().strip()
    msg = msg.rstrip("?.! ").strip()

    prefixes = (
        "what is ",
        "what are ",
        "what's ",
        "whats ",
        "define ",
        "explain ",
        "tell me about ",
        "describe ",
    )
    for prefix in prefixes:
        if msg.startswith(prefix):
            msg = msg[len(prefix):].strip()
            break

    msg = re.split(r",|\band\b|\bwhich\b|\bwhere\b|\bwhen\b|\bwho\b|\bwhy\b|\bhow\b", msg, maxsplit=1)[0].strip()
    msg = re.sub(r"^(a|an|the)\s+", "", msg).strip()
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]*", msg)
    return " ".join(words[:6]).strip()


def normalize_subject_key(subject: str) -> str:
    key = re.sub(r"[^a-z0-9\s\-]+", " ", subject.lower())
    key = re.sub(r"\s+", " ", key).strip()
    key = SUBJECT_ALIASES.get(key, key)
    return key


def get_canonical_definition(message: str) -> str | None:
    if not should_use_extractive_answer(message):
        return None

    subject = normalize_subject_key(extract_question_subject(message))
    if not subject:
        return None

    if subject in CANONICAL_DEFINITIONS:
        return CANONICAL_DEFINITIONS[subject]

    singular = subject[:-1] if subject.endswith("s") else subject
    singular = SUBJECT_ALIASES.get(singular, singular)
    return CANONICAL_DEFINITIONS.get(singular)


def build_guided_definition_user_message(message: str, canonical_definition: str) -> str:
    return (
        "Answer the question in one clear sentence. "
        "Use your own words, but stay faithful to the reference definition. "
        "Do not mention the reference explicitly.\n\n"
        f"Reference definition: {canonical_definition}\n\n"
        f"Question: {message}"
    )


def should_use_canonical_fallback(
    message: str,
    response: str,
    canonical_definition: str | None,
) -> bool:
    if not canonical_definition:
        return False

    cleaned = response.strip()
    if len(cleaned) < 32:
        return True

    lowered = cleaned.lower()
    if cleaned.endswith("..."):
        return True

    bad_phrases = (
        "ongoing debate",
        "no clear consensus",
        "subject of research",
        "nefarious purposes",
        "autonomous vehicles",
        "fraud detection",
        "predictive modeling",
    )
    if any(phrase in lowered for phrase in bad_phrases):
        return True

    subject_terms = [
        token
        for token in normalize_subject_key(extract_question_subject(message)).split()
        if token not in STOPWORDS
    ]
    if subject_terms and not all(token in lowered for token in subject_terms[:2]):
        return True

    return False


def finalize_model_response(
    message: str,
    response: str,
    canonical_definition: str | None = None,
) -> str:
    cleaned = response.strip()
    if should_use_canonical_fallback(message, cleaned, canonical_definition):
        return canonical_definition or cleaned
    return cleaned


def normalize_overlap_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]*", text.lower())


def is_question_echo(candidate: str, message: str) -> bool:
    candidate_text = candidate.strip()
    if candidate_text.endswith("?"):
        return True

    candidate_lower = candidate_text.lower()
    message_lower = message.strip().lower().rstrip("?.! ")

    if re.match(r"^(what|who|when|where|why|how)\b", candidate_lower):
        return True

    normalized_candidate = " ".join(normalize_overlap_text(candidate_lower))
    normalized_message = " ".join(normalize_overlap_text(message_lower))
    if normalized_candidate == normalized_message:
        return True

    return normalized_candidate.startswith(normalized_message + " ")


def _sentence_definition_score(sentence: str, subject: str, question_terms: list[str]) -> int:
    lowered = sentence.lower()
    score = 0

    if subject:
        if subject in lowered:
            score += 12
        if re.search(rf"\b{re.escape(subject)}\b\s+(is|are|was|were|refers to|means)\b", lowered):
            score += 24
        if lowered.startswith(subject):
            score += 8

    if re.search(r"\b(is|are|was|were|refers to|means)\s+(a|an|the)\b", lowered):
        score += 8

    score += 3 * sum(term in lowered for term in question_terms)

    if re.match(r"^(this|that|these|those|it|they)\b", lowered):
        score -= 10

    if sentence.endswith("..."):
        score -= 20

    return score


def build_extractive_answer(message: str, context: str) -> str | None:
    if not context.strip():
        return None

    normalized = re.sub(r"\[\d+\]", "", context)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    raw_sentences = re.split(r"(?<=[.!?])\s+", normalized)
    question_terms = extract_question_terms(message)
    subject = extract_question_subject(message)

    sentences: list[str] = []
    seen: set[str] = set()

    for sentence in raw_sentences:
        cleaned = sentence.strip(" -\n\t")
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" ,;:")
        if len(cleaned) < 35 or len(cleaned) > 240:
            continue

        alpha_ratio = sum(ch.isalpha() for ch in cleaned) / max(len(cleaned), 1)
        if alpha_ratio < 0.65:
            continue

        if cleaned.endswith("..."):
            continue

        if is_question_echo(cleaned, message):
            continue

        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        sentences.append(cleaned)

    if not sentences:
        return None

    ranked = sorted(
        sentences,
        key=lambda sentence: (
            -_sentence_definition_score(sentence, subject, question_terms),
            abs(len(sentence) - 110),
        ),
    )

    lead = ranked[0]
    if _sentence_definition_score(lead, subject, question_terms) < 8:
        return None

    answer = lead
    if answer[-1] not in ".!?":
        answer += "."
    return answer


def build_rag_user_message(message: str, context: str) -> str:
    snippet = re.sub(r"\s+", " ", context).strip()
    if len(snippet) > WEB_CONTEXT_CHARS:
        snippet = snippet[:WEB_CONTEXT_CHARS].rsplit(" ", 1)[0].rstrip(" ,.;:")
        snippet += "..."

    return (
        "Use the background below if it helps answer accurately. "
        "Answer in 1-2 short sentences. "
        "Do not repeat the question. "
        "Do not quote the background verbatim.\n\n"
        f"Background: {snippet}\n\n"
        f"Question: {message}\n\n"
        "Respond directly."
    )


def normalize_generation_settings(
    message: str,
    temperature: float | int | None,
    max_tokens: int | None,
) -> tuple[float, int]:
    factual = is_factual_question(message)
    requested_temp = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    requested_tokens = 150 if max_tokens is None else int(max_tokens)

    temp_cap = FACTUAL_TEMPERATURE_CAP if factual else CREATIVE_TEMPERATURE_CAP
    token_cap = FACTUAL_MAX_TOKENS if factual else CREATIVE_MAX_TOKENS

    clamped_temp = max(0.2, min(requested_temp, temp_cap))
    clamped_tokens = max(1, min(requested_tokens, token_cap))
    return clamped_temp, clamped_tokens


async def fetch_web_context(message: str) -> str | None:
    def _search() -> str | None:
        try:
            from ddgs import DDGS
        except Exception:
            return None

        try:
            with DDGS() as ddgs:
                snippets = []
                for result in ddgs.text(message, max_results=3):
                    body = str(result.get("body", "")).strip()
                    if body:
                        snippets.append(body)
                return " ".join(snippets) or None
        except Exception as exc:
            print(f"DuckDuckGo search failed: {exc}")
            return None

    return await asyncio.to_thread(_search)


async def prepare_model_payload(payload: dict[str, Any]) -> tuple[str | None, dict[str, Any], str]:
    message = str(payload.get("message", "")).strip()
    if not message:
        raise ValueError("message must not be empty")

    direct_response = try_solve_math(message)
    history = normalize_history(payload.get("history"))
    temperature, max_tokens = normalize_generation_settings(
        message,
        payload.get("temperature"),
        payload.get("max_tokens"),
    )

    body: dict[str, Any] = {
        "message": message,
        "history": history,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if direct_response is not None:
        return direct_response, body, "math"

    canonical_definition = get_canonical_definition(message)
    if canonical_definition is not None:
        body["message"] = build_guided_definition_user_message(message, canonical_definition)
        body["_canonical_answer"] = canonical_definition
        return None, body, "guided_definition"

    if is_factual_question(message):
        context = await fetch_web_context(message)
        if context:
            if should_use_extractive_answer(message):
                direct_answer = build_extractive_answer(message, context)
                if direct_answer:
                    return direct_answer, body, "extractive"
            body["message"] = build_rag_user_message(message, context)
            return None, body, "rag"
        return None, body, "factual"

    if is_creative_request(message):
        return None, body, "creative"

    return None, body, "chat"
