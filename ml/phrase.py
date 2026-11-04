"""Turns a list of recognised sign labels into a readable sentence."""

QUESTION_WORDS = {"where", "what", "who", "when", "why", "how"}
NEEDS_ARTICLE = {"doctor"}  # "I need a doctor"


def signs_to_sentence(labels: list[str]) -> str:
    words = [str(label).replace("_", " ").lower() for label in labels if str(label).strip()]
    if not words:
        return ""
    words = ["I" if w == "i" else w for w in words]
    if len(words) >= 3 and words[1] == "need" and words[2] in NEEDS_ARTICLE:
        words.insert(2, "a")
    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:]
    return sentence + ("?" if words[0].lower() in QUESTION_WORDS else ".")
