"""Prompt sets for benchmarking."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PROMPTS = [
    "Explain how a hash map works and when you would not use one.",
    "Write a short Python function that checks whether a string is a palindrome.",
    "Summarize the causes of the French Revolution in five bullet points.",
    "What is the difference between TCP and UDP? Give one use case for each.",
    "Describe the water cycle to a ten year old.",
    "Write a haiku about distributed systems.",
    "List three pros and three cons of remote work.",
    "Explain what a transformer neural network is in plain English.",
    "Give me a recipe for a quick vegetarian dinner using chickpeas.",
    "How does public key cryptography allow two strangers to communicate securely?",
    "Translate 'The weather is lovely today, let's go for a walk' into French and Spanish.",
    "Write a SQL query that returns the top 5 customers by total order value.",
    "What are the main differences between Kubernetes Deployments and StatefulSets?",
    "Explain the CAP theorem with a concrete example.",
    "Draft a polite email asking a colleague to review a pull request by Friday.",
    "Why is the sky blue? Answer in two short paragraphs.",
    "Compare Python lists and tuples, including performance considerations.",
    "Write a bash one-liner that counts unique IP addresses in an nginx access log.",
    "Describe three strategies for reducing latency in a web application.",
    "What is gradient descent and why does the learning rate matter?",
    "Give a step-by-step plan for learning to play the guitar in three months.",
    "Explain the difference between processes and threads.",
    "Write a short story opening line set on a space station.",
    "What is KV caching in large language model inference and why does it help?",
]


def load_prompts(path: str | None) -> list[str]:
    """Load prompts from a ``.jsonl`` (``{"prompt": ...}``) or plain-text (one per line) file."""
    if not path:
        return list(DEFAULT_PROMPTS)
    text = Path(path).read_text(encoding="utf-8")
    prompts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if path.endswith(".jsonl"):
            item = json.loads(line)
            prompts.append(item["prompt"] if isinstance(item, dict) else str(item))
        else:
            prompts.append(line)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts
