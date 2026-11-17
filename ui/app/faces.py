"""Whose faces are in a picture: Luna's, the owner's, or someone else's.

InsightFace's buffalo_l models, run on ONNX Runtime on the GPU beside the vision model (the CPU
with LUNASOCIA_VOICE_DEVICE=cpu, or when the GPU fails; about 0.2 s a picture there): SCRFD finds the faces and their five landmarks, and ArcFace turns each face, aligned
to 112 x 112, into a face print (512 numbers). The same person's prints point the same way
(cosine similarity about 0.6-0.9 for her pictures), different people's do not (0.3-0.5 for
women drawn to look like her): see model_evaluations/self_recognition.py. The vision model
judges faces by hair and jewellery and calls nearly any such woman "me", so it is told whose
faces these are instead. The models are downloaded to the data volume the first time; their
weights are for non-commercial use only.
"""

from __future__ import annotations

import gc
import io
import logging
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import SociaConfig
from app.model_usage import ModelUsage, measure_cuda_context, measuring

logger = logging.getLogger("socia")
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DETECTOR, RECOGNIZER = "det_10g.onnx", "w600k_r50.onnx"
DETECTION_SIDE = 640
DETECTION_SCORE = 0.5
OVERLAP = 0.4
# Where ArcFace expects the eyes, nose tip and mouth corners in its 112 x 112 input.
ARCFACE_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                           [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass
class Face:
    box: tuple[float, float, float, float]
    score: float
    print: np.ndarray


@dataclass
class Match:
    name: str | None
    similarity: float
    place: str


def similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The rotation, uniform scale and shift taking ``source`` points nearest to ``target`` (Umeyama),
    as a 2 x 3 matrix: how InsightFace aligns a face before ArcFace reads it."""
    mean_s, mean_t = source.mean(0), target.mean(0)
    s, t = source - mean_s, target - mean_t
    covariance = t.T @ s / len(source)
    u, d, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(covariance) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = np.trace(np.diag(d) @ sign) / s.var(0).sum()
    shift = mean_t - scale * rotation @ mean_s
    return np.hstack([scale * rotation, shift[:, None]]).astype(np.float32)


def overlap_filter(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Indices of the boxes to keep: the best-scored of each group that overlaps more than ``threshold``."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        w = np.maximum(0.0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]) + 1)
        h = np.maximum(0.0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]) + 1)
        shared = w * h / (areas[best] + areas[rest] - w * h)
        order = rest[shared <= threshold]
    return keep


def place(box: tuple[float, float, float, float], width: int) -> str:
    centre = (box[0] + box[2]) / 2 / width
    return "left" if centre < 0.4 else "right" if centre > 0.6 else "middle"


class Faces:
    def __init__(self, config: SociaConfig) -> None:
        self.config = config
        self.lock = threading.Lock()
        self.detector = self.recognizer = None
        # Known face prints by person, with the files (and their times) they were read from.
        self.known_prints: dict[str, tuple[tuple, np.ndarray]] = {}
        self.device = ""
        self.usage = ModelUsage("InsightFace buffalo_l (SCRFD + ArcFace)", "Knows her face and yours in photos",
                                lambda: (self.detector is not None, self.device if self.detector is not None
                                         else "loads with the vision model, for a photo"))

    @property
    def folder(self) -> Path:
        return self.config.models_dir / "insightface"

    def load(self) -> None:
        """The two models, downloaded once (buffalo_l, 280 MB, of which these are 180 MB)."""
        if self.detector is not None:
            return
        import onnxruntime

        if not all((self.folder / name).exists() for name in (DETECTOR, RECOGNIZER)):
            logger.info("Downloading the face models (InsightFace buffalo_l)")
            with urllib.request.urlopen(BUFFALO_URL, timeout=120) as reply:
                archive = zipfile.ZipFile(io.BytesIO(reply.read()))
            self.folder.mkdir(parents=True, exist_ok=True)
            for member in archive.namelist():
                name = Path(member).name
                if name in (DETECTOR, RECOGNIZER):
                    (self.folder / f"{name}.part").write_bytes(archive.read(member))
                    (self.folder / f"{name}.part").rename(self.folder / name)
        # Some layers run in a slower mode on GPUs newer than this build, and say so at every run.
        onnxruntime.set_default_logger_severity(3)
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if self.config.voice_device == "cuda":
            # Memory as each run needs it, not in growing blocks: it shares the GPU with the vision model.
            providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT",
                                                           "arena_extend_strategy": "kSameAsRequested"}))
        try:
            self.detector = onnxruntime.InferenceSession(str(self.folder / DETECTOR), options, providers=providers)
            self.recognizer = onnxruntime.InferenceSession(str(self.folder / RECOGNIZER), options, providers=providers)
        except Exception as error:  # noqa: BLE001 - the CPU reads faces too, only slower
            logger.warning("The face models could not use the GPU, using the CPU: %s", error)
            providers = ["CPUExecutionProvider"]
            self.detector = onnxruntime.InferenceSession(str(self.folder / DETECTOR), options, providers=providers)
            self.recognizer = onnxruntime.InferenceSession(str(self.folder / RECOGNIZER), options, providers=providers)
        self.device = "GPU (ONNX, CUDA)" if self.detector.get_providers()[0] == "CUDAExecutionProvider" else "CPU (ONNX)"

    def preload(self) -> None:
        """Download both models, load them and run each once, then let them go.

        The first run in a process compiles CUDA's kernels for this GPU (about 30 s, kept in
        CUDA_CACHE_PATH), which should not be the first photo's wait. They also measure their GPU memory
        here. Afterwards they stay off the GPU until a photo turn loads them beside the vision model.
        """
        try:
            import cv2

            if self.config.voice_device == "cuda":
                measure_cuda_context()
            blank = cv2.imencode(".jpg", np.full((256, 256, 3), 128, np.uint8))[1].tobytes()
            with measuring(self.usage):
                with self.lock:
                    self.load()
                self.read(blank)
                with self.lock:
                    aligned = np.zeros((1, 3, 112, 112), np.float32)
                    self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: aligned})
            logger.info("The face models are ready, on the %s (%s MB); they load again for each photo",
                        self.device, round(self.usage.vram_mb) if self.usage.vram_mb is not None else "?")
            self.unload()
        except Exception:  # noqa: BLE001 - loaded on first use instead
            logger.exception("Could not load the face models at start")

    def unload(self) -> None:
        """Free their GPU memory, after a photo turn; the next photo loads them again (about a second)."""
        with self.lock:
            if self.detector is None:
                return
            self.detector = self.recognizer = None
            gc.collect()

    # Finding and reading faces

    def detect(self, image: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Each face in a BGR image: its box, score and five landmarks (SCRFD, as InsightFace decodes it)."""
        import cv2

        height, width = image.shape[:2]
        if height / width > 1:
            new_h, new_w = DETECTION_SIDE, int(DETECTION_SIDE * width / height)
        else:
            new_w, new_h = DETECTION_SIDE, int(DETECTION_SIDE * height / width)
        scale = new_h / height
        canvas = np.zeros((DETECTION_SIDE, DETECTION_SIDE, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (DETECTION_SIDE, DETECTION_SIDE), (127.5, 127.5, 127.5),
                                     swapRB=True)
        outputs = self.detector.run(None, {self.detector.get_inputs()[0].name: blob})
        outputs = [o[0] if o.ndim == 3 else o for o in outputs]
        boxes, scores, points = [], [], []
        for level, stride in enumerate((8, 16, 32)):
            side = DETECTION_SIDE // stride
            centres = np.stack(np.mgrid[:side, :side][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            centres = np.repeat(centres, 2, axis=0)
            score = outputs[level].ravel()
            found = np.where(score >= DETECTION_SCORE)[0]
            distance = outputs[level + 3][found] * stride
            marks = outputs[level + 6][found] * stride
            at = centres[found]
            boxes.append(np.stack([at[:, 0] - distance[:, 0], at[:, 1] - distance[:, 1],
                                   at[:, 0] + distance[:, 2], at[:, 1] + distance[:, 3]], axis=-1))
            points.append(np.stack([at[:, i % 2] + marks[:, i] for i in range(10)], axis=-1).reshape(-1, 5, 2))
            scores.append(score[found])
        boxes, scores, points = np.vstack(boxes) / scale, np.concatenate(scores), np.vstack(points) / scale
        return [(boxes[i], float(scores[i]), points[i]) for i in overlap_filter(boxes, scores, OVERLAP)]

    def read(self, data: bytes) -> tuple[list[Face], int]:
        """The faces in a picture, largest first, with their prints; and the picture's width."""
        import cv2

        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return [], 0
        with self.lock, self.usage.record():
            self.load()
            faces = []
            for box, score, points in self.detect(image):
                aligned = cv2.warpAffine(image, similarity_transform(points, ARCFACE_POINTS), (112, 112), borderValue=0.0)
                blob = cv2.dnn.blobFromImages([aligned], 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
                vector = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: blob})[0][0]
                faces.append(Face(tuple(float(v) for v in box), score, vector / np.linalg.norm(vector)))
        faces.sort(key=lambda f: -(f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
        return faces, image.shape[1]

    # Who is who

    def prints_of(self, person: str, files: list[Path]) -> np.ndarray:
        """The largest face's print in each of a person's pictures, read again only when they change."""
        stamp = tuple((str(path), path.stat().st_mtime) for path in files if path.exists())
        cached = self.known_prints.get(person)
        if cached and cached[0] == stamp:
            return cached[1]
        prints = []
        for path, _ in stamp:
            faces, _ = self.read(Path(path).read_bytes())
            if faces:
                prints.append(faces[0].print)
        found = np.array(prints) if prints else np.zeros((0, 512), dtype=np.float32)
        self.known_prints[person] = (stamp, found)
        return found

    def who(self, data: bytes, people: dict[str, list[Path]]) -> list[Match]:
        """Each face in a picture with the person it is (or None), by the closest of their pictures."""
        faces, width = self.read(data)
        known = {name: self.prints_of(name, files) for name, files in people.items()}
        matches = []
        for face in faces:
            name, best = None, 0.0
            for person, prints in known.items():
                if len(prints):
                    score = float((prints @ face.print).max())
                    if score > best:
                        name, best = person, score
            matches.append(Match(name if best >= SociaConfig.FACE_MATCH else None, round(best, 2), place(face.box, width)))
        return matches


def faces_line(matches: list[Match], her: str, name: str) -> str:
    """What face recognition says about a picture, for the vision model and her later replies."""
    if not matches:
        return "Faces: none visible, so whoever it is cannot be told from a face: say you cannot see their face."
    parts = []
    for match in matches[:SociaConfig.FACES_NAMED]:
        where = f" ({match.place})" if len(matches) > 1 else ""
        if match.name == her:
            parts.append(f"you, {her}{where}")
        elif match.name:
            parts.append(f"{name}{where}")
        else:
            parts.append(f"someone who is neither you nor {name}{where}")
    more = len(matches) - SociaConfig.FACES_NAMED
    return "Faces: " + "; ".join(parts) + (f"; and {more} more" if more > 0 else "") + "."
