"""Rule-based complexity scorer: zero API calls, ~microseconds per query.

Each signal adds to / subtracts from a score. Score thresholds map to a tier, and
distance from the nearest threshold becomes the confidence -- borderline scores get
low confidence, which is what lets the hybrid classifier know when to ask an LLM.
"""
from __future__ import annotations

import re

from ..schemas import ClassificationResult, Complexity

SIMPLE_MAX = -0.15    # score <= this  -> SIMPLE
COMPLEX_MIN = 0.45    # score >= this  -> COMPLEX

_GREETING = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|thx|bye|ok|okay|good (morning|afternoon|evening|night))\b[\s!.,?]*(there|again)?[\s!.?]*$",
    re.I,
)
_LOOKUP = re.compile(
    r"^\s*(what('s| is| are| was) (the )?(capital|population|currency|meaning|definition|time|date)"
    r"|who (is|was|wrote|invented|discovered|painted)"
    r"|when (was|did|is)\b"
    r"|where (is|was|are)\b"
    r"|define\b|translate\b|convert\b"
    r"|what does .{1,30} mean"
    r"|how do you say\b"
    r"|is [\w\s.+#-]{1,30}\?$)",
    re.I,
)
_STRONG_KW = re.compile(
    r"\b(prove|derive|from scratch|trade-?offs?|time complexity|space complexity|optimi[sz]e|refactor|debug|"
    r"implement|algorithm|step[- ]by[- ]step|distributed|concurren\w*|race condition|dynamic programming|"
    r"architecture|design a (system|database|schema|service)|fault[- ]toleran\w*|consensus|migration)\b",
    re.I,
)
_MEDIUM_KW = re.compile(
    r"\b(analy[sz]e|compare|contrast|evaluate|explain why|pros and cons|summari[sz]e|critique|recommend|"
    r"strategy|difference between|how does|why does|write an? |create an? |build an? |plan)\b",
    re.I,
)
_CODE_BLOCK = re.compile(
    r"```|\bdef \w+\(|\bclass \w+|^\s*import \w+|\bSELECT\b[\s\S]{0,200}\bFROM\b|#include|public static|=>|\bfunction\s*\w*\(",
    re.I | re.M,
)
_CODE_REQUEST = re.compile(
    r"\b(write|create|build|implement|generate)\b[^.?!]{0,40}\b(function|script|program|class|query|regex|api|endpoint|module|code)\b",
    re.I,
)
_MATH = re.compile(
    r"\d+\s*[-+*/^=]\s*\d+|\b(integral|derivative|matrix|eigen\w*|theorem|equation|probability|polynomial|logarithm)\b|[∫∑√]",
    re.I,
)
_NUMBERED = re.compile(r"^\s*\d+[.)]\s", re.M)
_FORMAT_CONSTRAINT = re.compile(r"\b(at least|exactly|must|constraints?|requirements?|without using)\b", re.I)


class HeuristicClassifier:
    name = "heuristic"

    def score(self, query: str) -> tuple[float, dict[str, float]]:
        q = query.strip()
        words = len(q.split())
        s: dict[str, float] = {}

        # length
        if words <= 4:
            s["length_short"] = -0.3
        elif words <= 8:
            s["length_short"] = -0.15
        elif words >= 150:
            s["length_long"] = 0.35
        elif words >= 60:
            s["length_long"] = 0.25
        elif words >= 25:
            s["length_long"] = 0.1

        # trivial patterns (only when the query is short)
        if _GREETING.match(q):
            s["greeting"] = -0.5
        elif words <= 15 and _LOOKUP.match(q):
            s["simple_lookup"] = -0.4

        # keywords
        strong = len(set(m.group(0).lower() for m in _STRONG_KW.finditer(q)))
        medium = len(set(m.group(0).lower() for m in _MEDIUM_KW.finditer(q)))
        if strong:
            s["strong_keywords"] = min(0.3 * strong, 0.6)
        if medium:
            s["medium_keywords"] = min(0.15 * medium, 0.3)

        # code / math
        if _CODE_BLOCK.search(q):
            s["code_present"] = 0.4
        if _CODE_REQUEST.search(q):
            s["code_request"] = 0.25
        if _MATH.search(q):
            s["math"] = 0.3

        # multi-part asks
        if q.count("?") >= 3 or len(_NUMBERED.findall(q)) >= 3:
            s["multi_part"] = 0.2
        if _FORMAT_CONSTRAINT.search(q) and words >= 15:
            s["constraints"] = 0.1

        return sum(s.values()), s

    @staticmethod
    def _level_and_confidence(score: float) -> tuple[Complexity, float]:
        if score <= SIMPLE_MAX:
            level = Complexity.SIMPLE
        elif score >= COMPLEX_MIN:
            level = Complexity.COMPLEX
        else:
            level = Complexity.MODERATE
        distance = min(abs(score - SIMPLE_MAX), abs(score - COMPLEX_MIN))
        return level, round(min(0.95, 0.5 + distance), 3)

    def classify_sync(self, query: str) -> ClassificationResult:
        score, signals = self.score(query)
        score = round(score, 3)  # avoid float artefacts exactly at a threshold (0.6 - 0.15 = 0.4499...)
        level, confidence = self._level_and_confidence(score)
        return ClassificationResult(
            level=level, score=score, confidence=confidence, classifier=self.name, signals=signals
        )

    async def classify(self, query: str) -> ClassificationResult:
        return self.classify_sync(query)
