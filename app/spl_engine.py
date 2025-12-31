"""
SPL Decision Engine – Phase 2 (Layer 0: Reactive)

Layer 0 rules:
- Deterministic
- Zero cost
- Zero ambiguity
- Safe for voice agents
"""

import re
import string
from dataclasses import dataclass
from typing import Optional


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


@dataclass
class SPLResult:
    handled: bool
    response: Optional[str] = None
    layer: Optional[int] = None
    reason: Optional[str] = None


class SPLEngine:
    def __init__(self):
        self.min_length = 2

        # ===== Layer 0 =====
        self.filler_words = {"uh", "um", "hmm", "erm", "mm", "ah"}
        self.blocklist = {"fuck", "shit", "bitch", "asshole"}
        self.system_commands = {"repeat", "say that again", "hang up", "stop"}

        # ===== Layer 1 =====
        self.patterns = [
            {
                "name": "opening_time",
                "regex": r"(what (time|hours?)|when).*(open|opening|start|begin)|(open|opening|start).*(time|hours?)|(what are|what're|what's|whats).*(opening|open).*(hours?|time)",
                "response": "We open daily at 11:00 AM.",
                "confidence": 0.95,
            },
            {
                "name": "closing_time",
                "regex": r"(what (time|hours?)|when).*(close|closing|end|last order)|(close|closing|last order).*(time|hours?)|(what are|what're|what's|whats).*(closing|close).*(hours?|time)|(when is).*(last order|kitchen close)",
                "response": "We close daily at 10:30 PM, with last orders at 10:00 PM.",
                "confidence": 0.95,
            },
            {
                "name": "location",
                "regex": r"(where|location|address).*",
                "response": "We're located at MG Road, Bangalore.",
                "confidence": 0.95,
            },
            {
                "name": "menu",
                "regex": r"(menu|dishes|food|items)",
                "response": "We serve North Indian, South Indian, and Chinese cuisine.",
                "confidence": 0.9,
            },
            {
                "name": "greeting",
                "regex": r"^(hi|hello|hey)$",
                "response": "Hello! How can I help you?",
                "confidence": 0.9,
            },
            {
                "name": "thanks",
                "regex": r"(thanks|thank you)",
                "response": "You're welcome!",
                "confidence": 0.9,
            },
        ]

    def decide(self, text: str) -> SPLResult:
        normalized = normalize_text(text)

        # =========================
        # Layer 0.1 – Numeric-only
        # =========================
        if normalized.isdigit():
            print("[SPL:L0] Rejected: numeric-only input")
            return SPLResult(
                handled=True,
                response="I'm not sure how to respond to numbers alone. Could you provide more context?",
                layer=0,
                reason="Numeric-only input"
            )

        # =========================
        # Layer 0.2 – Empty / Noise
        # =========================
        if len(normalized) < self.min_length:
            print("[SPL:L0] Rejected: too short / noise")
            return SPLResult(
                handled=True,
                response="Sorry, I didn't catch that.",
                layer=0,
                reason="Input too short",
            )

        # =========================
        # Layer 0.2 – Filler words
        # =========================
        filler_clean = "".join(c for c in normalized if c.isalpha())

        if filler_clean in self.filler_words:
            print("[SPL:L0] Suppressed: filler utterance")
            return SPLResult(
                handled=True,
                response="Yes?",
                layer=0,
                reason="Filler utterance",
            )

        # =========================
        # Layer 0.3 – Profanity
        # =========================
        for bad_word in self.blocklist:
            if bad_word in normalized:
                print("[SPL:L0] Blocked: profanity detected")
                return SPLResult(
                    handled=True,
                    response="Let's keep things respectful.",
                    layer=0,
                    reason="Profanity detected",
                )

        # =========================
        # Layer 0.4 – System commands
        # =========================
        for command in self.system_commands:
            if command in normalized:
                print("[SPL:L0] System command detected → passing")
                return SPLResult(
                    handled=False,
                    layer=0,
                    reason="System command detected",
                )

        # =========================
        # Layer 0 fallback
        # =========================
        print("[SPL:L0] No decision → passing to Layer 1")

        # =========================
        # Layer 1 – Pattern matching
        # =========================
        for pattern in self.patterns:
            if re.search(pattern["regex"], normalized):
                print(f"[SPL:L1] Matched pattern: {pattern['name']}")
                return SPLResult(
                    handled=True,
                    response=pattern["response"],
                    layer=1,
                    reason=f"Pattern match: {pattern['name']}",
                )

        # =========================
        # Final fallback
        # =========================
        print("[SPL:L1] No decision → passing to LLM")
        return SPLResult(
            handled=False,
            layer=1,
            reason="No pattern matched",
        )
