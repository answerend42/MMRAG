"""CLIP 向量编码器。"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def select_device(device: str | None = None) -> str:
    """! @brief 解析推理设备，支持 auto、cpu、cuda、cuda:N 和 mps。"""

    requested = (device or "auto").strip().lower()
    if requested in {"", "auto"}:
        return select_auto_device()
    if requested == "cpu":
        return "cpu"
    try:
        import torch
    except Exception as exc:
        raise ValueError(f"device {requested!r} requires PyTorch to be importable") from exc
    if requested == "mps":
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        raise ValueError("MPS requested but torch.backends.mps.is_available() is False")
    if requested == "cuda" or requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is False")
        if requested.startswith("cuda:"):
            suffix = requested.split(":", 1)[1]
            if not suffix.isdigit():
                raise ValueError("CUDA device must use the form cuda or cuda:N")
            index = int(suffix)
            device_count = torch.cuda.device_count()
            if index >= device_count:
                raise ValueError(f"CUDA device index {index} is out of range; found {device_count}")
        return requested
    raise ValueError("device must be one of auto, cpu, cuda, cuda:N, or mps")


def select_auto_device() -> str:
    """! @brief 自动选择可用设备，优先级为 MPS、CUDA、CPU。"""

    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _is_chinese_clip(model_name: str) -> bool:
    """! @brief 判断模型是否为 Chinese-CLIP 系列。"""
    name = model_name.lower().replace("-", "").replace("_", "").replace("/", "")
    return "chineseclip" in name


class ClipEmbedder:
    """! @brief 使用 Hugging Face CLIPModel 生成图片和文本向量。

    支持标准 CLIP（openai/clip-*）和 Chinese-CLIP（OFA-Sys/Chinese-CLIP-*）
    两类模型，根据模型名自动选择。
    """

    def __init__(
        self,
        model_name: str = "data/models/openai_clip-vit-base-patch32",
        device: str | None = None,
    ):
        """! @brief 加载 CLIP 模型，并把模型放到自动选择的设备上。"""

        import torch

        self.torch = torch
        self.model_name = model_name
        self.device = torch.device(select_device(device))

        if _is_chinese_clip(model_name):
            from transformers import ChineseCLIPModel, ChineseCLIPProcessor

            self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
            self.model = ChineseCLIPModel.from_pretrained(model_name)
        else:
            from transformers import CLIPModel, CLIPProcessor

            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model = CLIPModel.from_pretrained(model_name)

        self.model.eval()
        self.model.to(self.device)

    def encode_images(self, images: list[Image.Image], batch_size: int = 16) -> np.ndarray:
        """! @brief 批量编码图片，并返回 float32 L2 归一化矩阵。"""

        if not images:
            return np.empty((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = [image.convert("RGB") for image in images[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            batches.append(self._encode_inputs(inputs, "image"))
        return l2_normalize(np.vstack(batches))

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """! @brief 批量编码文本，并返回 float32 L2 归一化矩阵。"""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.processor(text=batch, padding=True, truncation=True, return_tensors="pt")
            batches.append(self._encode_inputs(inputs, "text"))
        return l2_normalize(np.vstack(batches))

    def _encode_inputs(self, inputs: dict, kind: Literal["image", "text"]) -> np.ndarray:
        """! @brief 在当前设备执行一次 batch；MPS 失败时降级到 CPU 重跑。"""

        try:
            return self._forward(inputs, kind)
        except Exception as exc:
            if self.device.type != "mps":
                raise
            logger.warning("MPS inference failed, falling back to CPU: %s", exc)
            self.device = self.torch.device("cpu")
            self.model.to(self.device)
            return self._forward(inputs, kind)

    def _forward(self, inputs: dict, kind: Literal["image", "text"]) -> np.ndarray:
        """! @brief 调用 CLIP 图片或文本特征接口。"""

        tensors = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            if kind == "image":
                features = self.model.get_image_features(**tensors)
            else:
                features = self.model.get_text_features(**tensors)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        return features.detach().cpu().numpy().astype(np.float32)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """! @brief 对向量矩阵按行做 L2 归一化。"""

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)
