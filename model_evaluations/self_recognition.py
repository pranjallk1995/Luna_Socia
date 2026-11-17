"""Does Luna know her own face, and the owner's? Her real photo turn, on photos of her and of others.

A folder of photos, kept out of git (your photos of her and of you are personal):

    <folder>/her/          photos of her
    <folder>/others/       photos of other people, best drawn to look like her (same hair, outfits)
    <folder>/owner/        optional: photos of the owner, not among the photos in "Photos of you"
    <folder>/same.txt      optional: "photo.png reference.jpg" per line, for photos that are one of
                           her own pictures, so that one is left out when its face is matched

Each photo goes, in a fresh chat, to the vision model exactly as in Socia: her prompt with SEEING,
her profile picture, and the photo with its "Faces:" line from face recognition. She is asked
"Who is this?" and "Is this you?" (twice each); the chat model then sorts each reply into YES (she
says it is her), NO or UNSURE. Face recognition's own scores are kept too. Without --faces, the
"Faces:" line is left out, as before Socia 0.9.

Run it beside the running Socia (in the notebooks' container, which has her code, data and GPU):
    python model_evaluations/self_recognition.py <folder> [--no-faces] [--repeat 2]
It writes <folder>/self_recognition[_no_faces].jsonl and prints a summary.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from app import chat as socia_chat
from app.config import SociaConfig
from app.faces import Faces, faces_line
from app.ollama import Ollama
from app.persona import TOOLS, system_prompt
from app.store import SociaStore

QUESTIONS = ["Who is this?", "Is this you?"]
JUDGE = """Luna was shown a photo and asked "{question}". She replied:

"{reply}"

Does Luna say the woman in the photo is herself (Luna)? Answer with one word: YES (she says it is her), NO (she says
it is someone else, or not her), or UNSURE (she does not say, or hedges)."""


def photos(folder: Path) -> list[tuple[Path, str]]:
    kinds = {"her": "her", "others": "other", "lookalikes": "other", "owner": "owner"}
    return [(path, kind) for sub, kind in kinds.items() if (folder / sub).is_dir()
            for path in sorted((folder / sub).iterdir()) if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--no-faces", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--companion", type=int, default=1)
    parser.add_argument("--name", default="Pranjall", help="the owner's name, as she calls them")
    args = parser.parse_args()
    NAME = args.name
    config = SociaConfig.from_env()
    store = SociaStore(config.database_url)
    ollama = Ollama(config)
    faces = Faces(config)
    conversation = socia_chat.Conversation(config, store, ollama, None, faces=None if args.no_faces else faces)
    her = store.companion(args.companion)
    looks = config.profile_path(her["id"]).with_suffix(".txt")
    prompt = system_prompt(her, NAME, datetime(2026, 11, 17, 21, 0), looks.read_text() if looks.exists() else "")
    prompt += socia_chat.SEEING.format(name=NAME)
    profile = base64.b64encode(socia_chat.seen_size(config.profile_path(her["id"]).read_bytes(),
                                                    SociaConfig.PHOTO_SEEN_SIDE // 2)).decode()
    same = {}
    if (args.folder / "same.txt").exists():
        same = dict(line.split()[:2] for line in (args.folder / "same.txt").read_text().splitlines() if line.strip())
    her_pictures = [config.profile_path(her["id"]), *config.references(her["id"])]

    records = []
    for path, kind in photos(args.folder):
        data = path.read_bytes()
        # Face recognition's scores, against her pictures but the one this photo is.
        people = {her["name"]: [p for p in her_pictures if p.name != same.get(path.name)], NAME: config.owner_photos()}
        matches = faces.who(data, people)
        line = "" if args.no_faces else faces_line(matches, her["name"], NAME)
        seen = base64.b64encode(socia_chat.seen_size(data, SociaConfig.PHOTO_SEEN_SIDE)).decode()
        for question in QUESTIONS:
            for round_ in range(args.repeat):
                label = f'the photo {NAME} sent just now with "{question}"' + (f". {line}" if line else "")
                content = f"{question}\n[Pictures you see now: 1) your profile picture: this is you, your face; 2) {label}]"
                messages = [{"role": "system", "content": prompt},
                            {"role": "user", "content": content, "images": [profile, seen]}]
                start, said = time.monotonic(), ""
                async for piece in ollama.chat(messages, TOOLS, seeing=True):
                    said += piece.get("text", "")
                records.append({"photo": f"{path.parent.name}/{path.name}", "kind": kind, "question": question,
                                "round": round_, "faces": line, "matches": [m.__dict__ for m in matches],
                                "reply": " ".join(said.split()), "seconds": round(time.monotonic() - start, 1)})
                print(records[-1]["photo"], question, "|", records[-1]["reply"][:100], flush=True)
    for record in records:
        answer = await ollama.complete(JUDGE.format(question=record["question"], reply=record["reply"] or "(no words)"))
        record["claims_her"] = next((w for w in ("YES", "NO", "UNSURE") if w in answer.upper()), "UNSURE")
    await ollama.back_to_chat()
    out = args.folder / ("self_recognition_no_faces.jsonl" if args.no_faces else "self_recognition.jsonl")
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    print(f"\n{out.name}: does she say it is her?")
    for kind in ("her", "other", "owner"):
        rows = [r for r in records if r["kind"] == kind]
        if rows:
            counts = Counter(r["claims_her"] for r in rows)
            print(f"  {kind:>5}: yes {counts['YES']}, no {counts['NO']}, unsure {counts['UNSURE']} of {len(rows)}")
    print("Face recognition (best similarity to her pictures, per photo):")
    for path, kind in photos(args.folder):
        row = next(r for r in records if r["photo"] == f"{path.parent.name}/{path.name}")
        scores = [m["similarity"] for m in row["matches"]]
        print(f"  {kind:>5} {row['photo']:<24} {', '.join(f'{s:.2f}' for s in scores) or 'no face'}  {row['faces']}")
    print("median reply", np.median([r["seconds"] for r in records]), "s")


if __name__ == "__main__":
    asyncio.run(main())
