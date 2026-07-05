from __future__ import annotations

RATIO_SUFFIXES = ("_ratio", "_share")


def format_percent(value: float, *, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


def format_probability(probs: dict[str, float], *, decimals: int = 2) -> dict[str, str]:
    return {label: format_percent(prob, decimals=decimals) for label, prob in probs.items()}


def format_feature_value(name: str, value: float, *, decimals: int = 2) -> str:
    if any(name.endswith(suffix) for suffix in RATIO_SUFFIXES):
        return format_percent(float(value), decimals=decimals)
    if name.endswith("_zscore"):
        return f"{float(value):.{decimals}f}"
    if name == "keyword_score":
        return f"{float(value):.{decimals}f}"
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.{decimals}f}"


def format_metric(name: str, value: float) -> str:
    if name == "accuracy":
        return format_percent(value, decimals=2)
    if name == "log_loss":
        return f"{value:.4f}"
    return f"{value:.4f}"
