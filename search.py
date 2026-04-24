"""Core semantic image search engine using SigLIP 2 + FAISS."""
from __future__ import annotations
import gc
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import faiss


class ImageSearchEngine:
    """Build and query a semantic image index."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-256",
        device: str | None = None,
        batch_size: int = 8,  # ← tune down to 4 if still freezing
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            AutoModel.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = batch_size
        self.index: faiss.Index | None = None
        self.paths: List[str] = []

    # ---------- encoders ----------
    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        vectors = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = self.model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0]  # CLS token
            feats = feats / feats.norm(dim=-1, keepdim=True)
            vectors.append(feats.cpu().numpy())
            # free memory after each batch
            del inputs, feats, batch
            gc.collect()
        return np.vstack(vectors).astype("float32")

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, padding="max_length", return_tensors="pt"
        ).to(self.device)
        feats = self.model.get_text_features(**inputs)
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        elif hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state[:, 0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype("float32")

    # ---------- indexing ----------
    def build_index(
        self,
        image_dir: str,
        exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".bmp"),
    ) -> int:
        paths = sorted(
            str(p) for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts
        )
        if not paths:
            raise ValueError(f"No images found in {image_dir}")

        all_embs = []
        good_paths = []

        for i in range(0, len(paths), self.batch_size):
            batch_paths = paths[i : i + self.batch_size]
            images = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    images.append(img)
                    good_paths.append(p)
                except Exception as e:
                    print(f"Skip {p}: {e}")

            if not images:
                continue

            embs = self.encode_images(images)
            all_embs.append(embs)

            # discard images immediately after encoding
            del images, embs
            gc.collect()

            print(f"Indexed {len(good_paths)}/{len(paths)} images...", end="\r")

        final_embs = np.vstack(all_embs).astype("float32")
        self.index = faiss.IndexFlatIP(final_embs.shape[1])
        self.index.add(final_embs)
        self.paths = good_paths
        return len(good_paths)

    def save(self, out_dir: str) -> None:
        assert self.index is not None, "Build the index first."
        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "paths.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.paths))

    def load(self, out_dir: str) -> None:
        self.index = faiss.read_index(os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "paths.txt"), encoding="utf-8") as f:
            self.paths = [line.strip() for line in f if line.strip()]

    # ---------- querying ----------
    def search_text(self, query: str, k: int = 9) -> List[Tuple[str, float]]:
        assert self.index is not None, "Load or build the index first."
        q = self.encode_text([query])
        scores, idxs = self.index.search(q, k)
        return [(self.paths[i], float(s)) for i, s in zip(idxs[0], scores[0])]

    def search_image(
        self, image: Union[str, Image.Image], k: int = 9
    ) -> List[Tuple[str, float]]:
        assert self.index is not None, "Load or build the index first."
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        q = self.encode_images([image])
        scores, idxs = self.index.search(q, k)
        return [(self.paths[i], float(s)) for i, s in zip(idxs[0], scores[0])]