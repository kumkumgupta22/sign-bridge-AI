import json

from . import config


def load_vocabulary() -> dict:
    with open(config.VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)


def labels() -> list[str]:
    return [s["label"] for s in load_vocabulary()["signs"]]


def find_sign(label: str) -> dict | None:
    for s in load_vocabulary()["signs"]:
        if s["label"].upper() == label.upper():
            return s
    return None
