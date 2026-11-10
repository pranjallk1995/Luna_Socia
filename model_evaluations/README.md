# Model evaluations

Compare two chat models on a fixed set of conversations before changing
`LUNASOCIA_CHAT_MODEL`. `scenarios.json` holds them: everyday texts, a question about
being an AI, a request for code, a message in Hindi, a photo to react to, and requests
for a selfie, an outfit, a view, and Luna added to a photo. Each reply goes through
Socia's own Ollama client, system prompt (for "Sam", at a fixed time), `create_image`
tool and settings; nothing is stored and no picture is made. Replies are sampled at
Socia's temperature, so every scenario runs three times (`--repeat`).

The scenarios are made up and hold no personal data; results are still kept out of git
in `results/` (see `.gitignore`).

## What is scored

| Measure | What it shows |
| --- | --- |
| `tool_right` | `create_image` called when a picture is asked for, and only then |
| `arguments_right` | Its call has a description, and `include_luna` and `use_their_photo` set as the request needs |
| `language_right` | A reply in Hindi (Devanagari) to a message in Hindi |
| `on_topic` | A reply that picks up what was said earlier (the driving test) or what the photo shows |
| `sounds_like_ai` | Admits to being an AI or assistant, offers help like a service, or writes code |
| `list_or_essay` | More than three sentences, or a list |
| `too_many_emoji` | More than three emoji |
| `empty`, `thinking_shown`, `failures` | No reply at all, reasoning shown, or a failed call |
| `scenarios_with_misses` | Which scenarios each model got wrong, and how often |

Whether she still sounds like Luna is for a person to judge: `report.md` lists every
reply of both models side by side, with what each one missed.

## Running one

```sh
C=luna_socia-ui-1
docker exec $C mkdir -p /tmp/socia_eval
for f in run.py score.py scenarios.json; do docker cp model_evaluations/$f $C:/tmp/socia_eval/; done
docker exec -e PYTHONPATH=/workspace/ui $C python /tmp/socia_eval/run.py old qwen3:14b /tmp/socia_eval/old.jsonl
docker exec -e PYTHONPATH=/workspace/ui $C python /tmp/socia_eval/run.py new <model> /tmp/socia_eval/new.jsonl
docker exec $C python /tmp/socia_eval/score.py /tmp/socia_eval
docker cp $C:/tmp/socia_eval model_evaluations/results/<date>_<old>-vs-<new>
```

The chat model stays resident on the GPU, so it must fit there at
`SENTIA_RESIDENT_CONTEXT`: run Luna Sentia's `model_checks/check_model.py <model>
--context 8192` first. Photo descriptions (`LUNASOCIA_VISION_MODEL`) are not scored
here; check a few photos by hand after changing that model.
