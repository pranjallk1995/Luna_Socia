"""Which of her other pictures go with a picture of her: the freshest ones, so her poses do not repeat.

Her profile picture goes with every picture of her (her face, and the picture's shape). Of her other pictures, up
to SociaConfig.MAX_REFERENCES of them, the PICTURE_REFERENCES best-scored go too. A picture's score is her
memory's ACT-R base-level activation turned round (recall.strength: ln(sum of age ** -0.5) over the times it was
used, in days), so the more often and the more recently one was used, the lower it scores, and it recovers with
time. Each picture the owner liked (LIKED_REACTIONS on it) that used it adds a little, LIKE_WEIGHT * ln(1 + likes),
so poses they like come back sooner. One never used scores highest; a little noise settles ties.
"""

from __future__ import annotations

import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import SociaConfig
from app.recall import strength

# Never used: above any used picture, whatever its likes.
UNUSED = 99.0
TIE_NOISE = 0.05


def score(uses: list[datetime], likes: int, now: datetime) -> float:
    """Higher the longer since a picture was used and the less often, and a little higher for each liked picture."""
    tired = strength(uses, now) if uses else -UNUSED
    return -tired + SociaConfig.LIKE_WEIGHT * math.log1p(likes)


def chosen(pictures: list[Path], history: dict[str, dict[str, Any]], now: datetime, count: int,
           rng: random.Random | None = None) -> list[Path]:
    """The ``count`` best-scored of ``pictures``; ``history`` maps a picture's name to its uses and likes."""
    rng = rng or random.Random()
    scored = {}
    for picture in pictures:
        past = history.get(picture.name) or {}
        scored[picture] = score(past.get("uses") or [], past.get("likes") or 0, now) + rng.uniform(0, TIE_NOISE)
    return sorted(pictures, key=lambda picture: -scored[picture])[:count]
