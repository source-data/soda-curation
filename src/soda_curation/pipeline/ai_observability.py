"""Shared helpers for safe AI payload logging."""

from typing import Any, Dict, List, Optional


def safe_excerpt(text: Optional[str], max_chars: int = 160) -> str:
    """Return a compact single-line excerpt suitable for logs."""
    if not text:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars]}..."


def summarize_text(text: Optional[str], max_excerpt_chars: int = 160) -> Dict[str, Any]:
    """Summarize text without logging the full content."""
    value = text or ""
    return {
        "length": len(value),
        "excerpt": safe_excerpt(value, max_chars=max_excerpt_chars),
    }


def summarize_messages(
    messages: List[Dict[str, Any]], max_excerpt_chars: int = 160
) -> Dict[str, Any]:
    """
    Summarize a chat payload for observability.

    The summary is designed to confirm what reached the AI API without logging
    complete prompt bodies.
    """
    summary: Dict[str, Any] = {
        "message_count": len(messages),
        "text_char_count": 0,
        "image_count": 0,
        "image_base64_char_count": 0,
        "first_user_excerpt": "",
    }

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            summary["text_char_count"] += len(content)
            if role == "user" and not summary["first_user_excerpt"]:
                summary["first_user_excerpt"] = safe_excerpt(
                    content, max_chars=max_excerpt_chars
                )
            continue

        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text", "")
                summary["text_char_count"] += len(text)
                if role == "user" and not summary["first_user_excerpt"]:
                    summary["first_user_excerpt"] = safe_excerpt(
                        text, max_chars=max_excerpt_chars
                    )
            elif item_type == "image_url":
                summary["image_count"] += 1
                image_url = item.get("image_url", {}).get("url", "")
                if isinstance(image_url, str) and "," in image_url:
                    summary["image_base64_char_count"] += len(
                        image_url.split(",", 1)[1]
                    )

    return summary
