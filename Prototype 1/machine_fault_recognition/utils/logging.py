from __future__ import annotations


def banner(title: str, subtitle: str = "") -> None:
    width = 78
    print(f"\n{'═' * width}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'═' * width}")


def section(text: str) -> None:
    print(f"\n  ▶ {text}")


def ok(text: str) -> None:
    print(f"    ✓ {text}")


def info(text: str) -> None:
    print(f"    • {text}")
