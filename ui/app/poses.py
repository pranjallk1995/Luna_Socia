"""Which of her other pictures go with a picture of her: drawn at random, the ones used most lately least likely.

Her profile picture goes with every picture of her (her face). Of her other pictures, up to
SociaConfig.MAX_REFERENCES of them, PICTURE_REFERENCES are drawn at random. Each has a score: every time it is used
its score goes up by 1 / sqrt(days since), so it rises with each use and wears off with time, like her memory's
ACT-R base-level activation (recall.strength). The higher the score, the less likely it is drawn: its weight is
1 / (1 + score), so one never used weighs 1, one used an hour ago about 0.17, one used a week ago about 0.73.
Each picture the owner liked (LIKED_REACTIONS on it) that used it raises the weight by LIKE_WEIGHT * ln(1 + likes),
so poses they like come back sooner.
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import SociaConfig
from app.recall import strength


def score(uses: list[datetime], now: datetime) -> float:
    """How much a picture was used: higher the more often and the more lately; 0 when never."""
    return math.exp(strength(uses, now)) if uses else 0.0


def weight(uses: list[datetime], likes: int, now: datetime) -> float:
    """Its chance to be drawn, against the others'."""
    return (1 + SociaConfig.LIKE_WEIGHT * math.log1p(likes)) / (1 + score(uses, now))


def chosen(pictures: list[Path], history: dict[str, dict[str, Any]], now: datetime, count: int,
           rng: random.Random | None = None) -> list[Path]:
    """``count`` of ``pictures`` drawn at random by weight, with no repeats; ``history`` maps a picture's name to
    its uses and likes. Using them (Store.add_picture_references) is what raises their scores."""
    rng = rng or random.Random()
    weights = {picture: weight((history.get(picture.name) or {}).get("uses") or [],
                               (history.get(picture.name) or {}).get("likes") or 0, now) for picture in pictures}
    # Weighted draws without repeats (Efraimidis and Spirakis): the largest random ** (1 / weight).
    keys = {picture: rng.random() ** (1 / weights[picture]) for picture in pictures}
    return sorted(pictures, key=lambda picture: -keys[picture])[:count]
