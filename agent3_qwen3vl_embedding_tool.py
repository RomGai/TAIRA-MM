from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class Agent3Qwen3VLEmbeddingTool:
    """Incremental multimodal embedding cache for Agent3 similarity recall.

    - Item embedding input supports both text and image.
    - Query embedding input supports text-only.
    - Cache is append-only by missing item_id and reusable across runs.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        cache_path: str | Path = "./cache/agent3_qwen3_vl_embedding_cache.npz",
        batch_size: int = 8,
        torch_dtype: Any | None = None,
        attn_implementation: str | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.cache_path = Path(cache_path)
        self.batch_size = max(1, int(batch_size))
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self._embedder = None

    def _load_embedder(self):
        if self._embedder is not None:
            return self._embedder

        try:
            from qwen3_vl_embedding import Qwen3VLEmbedder
        except ModuleNotFoundError:
            from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

        kwargs: Dict[str, Any] = {}
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = self.torch_dtype
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        self._embedder = Qwen3VLEmbedder(model_name_or_path=self.model_name_or_path, **kwargs)
        return self._embedder

    def _load_cache(self) -> tuple[List[str], np.ndarray | None]:
        if not self.cache_path.exists():
            return [], None
        npz = np.load(self.cache_path, allow_pickle=True)
        item_ids = [str(x) for x in npz["item_ids"].tolist()]
        embeddings = npz["item_embeddings"].astype(np.float32, copy=False)
        return item_ids, embeddings

    def _save_cache(self, item_ids: List[str], item_embeddings: np.ndarray) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            item_ids=np.array(item_ids),
            item_embeddings=item_embeddings.astype(np.float32, copy=False),
        )

    @staticmethod
    def _l2_normalize(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    def _embed_inputs(self, inputs: List[Dict[str, Any]]) -> np.ndarray:
        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)

        model = self._load_embedder()
        chunks: List[np.ndarray] = []
        for start in range(0, len(inputs), self.batch_size):
            end = min(len(inputs), start + self.batch_size)
            emb = model.process(inputs[start:end])
            chunk_np = emb.detach().float().cpu().numpy().astype(np.float32, copy=False)
            chunks.append(chunk_np)
        return np.concatenate(chunks, axis=0)

    def build_or_update_item_embedding_cache(self, item_payloads: Dict[str, Dict[str, Any]]) -> tuple[List[str], np.ndarray]:
        """Embed only missing item_ids, then persist merged cache.

        item_payloads[item_id] = {"text": str, "image": str(optional)}
        """
        cached_item_ids, cached_embeddings = self._load_cache()
        cached_set = set(cached_item_ids)

        missing_ids = [iid for iid in item_payloads.keys() if iid not in cached_set]
        if not missing_ids:
            if cached_embeddings is None:
                return [], np.zeros((0, 0), dtype=np.float32)
            return cached_item_ids, cached_embeddings

        inputs: List[Dict[str, Any]] = []
        for iid in missing_ids:
            payload = item_payloads.get(iid, {})
            obj: Dict[str, Any] = {}
            text = str(payload.get("text", "") or "").strip()
            image = str(payload.get("image", "") or "").strip()
            if text:
                obj["text"] = text
            if image:
                obj["image"] = image
            inputs.append(obj)

        new_emb = self._embed_inputs(inputs)

        if cached_embeddings is None:
            merged_ids = list(missing_ids)
            merged_emb = new_emb
        else:
            merged_ids = list(cached_item_ids) + list(missing_ids)
            merged_emb = np.concatenate([cached_embeddings, new_emb], axis=0)

        self._save_cache(merged_ids, merged_emb)
        return merged_ids, merged_emb

    def embed_query_texts(self, queries: Sequence[str]) -> np.ndarray:
        inputs = [{"text": str(q or "")} for q in queries]
        return self._embed_inputs(inputs)

    def rank_items_by_query(
        self,
        query: str,
        item_ids: Sequence[str],
        item_embeddings: np.ndarray,
        topk: int,
    ) -> List[str]:
        if not item_ids or item_embeddings.size == 0:
            return []
        q_emb = self.embed_query_texts([query])
        q_norm = self._l2_normalize(q_emb)[0]
        item_norm = self._l2_normalize(item_embeddings)
        sims = item_norm @ q_norm
        indices = np.argsort(-sims)[: max(1, int(topk))]
        return [str(item_ids[int(i)]) for i in indices]


class Agent3TextEmbeddingTool:
    """Incremental text embedding cache for Agent3 (Qwen3-Embedding-0.6B path)."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        cache_path: str | Path = "./cache/agent3_item_embedding_cache.npz",
        batch_size: int = 64,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.cache_path = Path(cache_path)
        self.batch_size = max(1, int(batch_size))
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name_or_path)
        return self._model

    def _load_cache(self) -> tuple[List[str], np.ndarray | None]:
        if not self.cache_path.exists():
            return [], None
        npz = np.load(self.cache_path, allow_pickle=True)
        item_ids = [str(x) for x in npz["item_ids"].tolist()]
        embeddings = npz["item_embeddings"].astype(np.float32, copy=False)
        return item_ids, embeddings

    def _save_cache(self, item_ids: List[str], item_embeddings: np.ndarray) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.cache_path,
            item_ids=np.array(item_ids),
            item_embeddings=item_embeddings.astype(np.float32, copy=False),
        )

    def _encode_texts(self, texts: List[str], prompt_name: str | None) -> np.ndarray:
        model = self._load_model()
        if torch is not None:
            with torch.inference_mode():
                emb = model.encode(
                    texts,
                    batch_size=self.batch_size,
                    prompt_name=prompt_name,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                return emb.astype(np.float32, copy=False)
        emb = model.encode(
            texts,
            batch_size=self.batch_size,
            prompt_name=prompt_name,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def build_or_update_item_embedding_cache(self, item_texts: Dict[str, str]) -> tuple[List[str], np.ndarray]:
        cached_item_ids, cached_embeddings = self._load_cache()
        cached_set = set(cached_item_ids)
        missing_ids = [iid for iid in item_texts.keys() if iid not in cached_set]

        if not missing_ids:
            if cached_embeddings is None:
                return [], np.zeros((0, 0), dtype=np.float32)
            return cached_item_ids, cached_embeddings

        missing_texts = [str(item_texts.get(iid, "") or "") for iid in missing_ids]
        new_emb = self._encode_texts(missing_texts, prompt_name=None)

        if cached_embeddings is None:
            merged_ids = list(missing_ids)
            merged_emb = new_emb
        else:
            merged_ids = list(cached_item_ids) + list(missing_ids)
            merged_emb = np.concatenate([cached_embeddings, new_emb], axis=0)

        self._save_cache(merged_ids, merged_emb)
        return merged_ids, merged_emb

    def embed_query_texts(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode_texts([str(q or "") for q in queries], prompt_name="query")


def save_agent3_qwen3vl_embedding_cache_manifest(
    output_path: str | Path,
    *,
    model_name_or_path: str,
    cache_path: str | Path,
    item_count: int,
) -> None:
    payload = {
        "tool": "Agent3Qwen3VLEmbeddingTool",
        "model_name_or_path": model_name_or_path,
        "cache_path": str(cache_path),
        "item_count": int(item_count),
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
