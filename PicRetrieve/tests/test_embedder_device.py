import sys
from types import SimpleNamespace

import pytest

from app.embedder import select_device


class _FakeMps:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeCuda:
    def __init__(self, available: bool, count: int = 0):
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


def install_fake_torch(monkeypatch, *, cuda: bool, mps: bool, cuda_count: int = 0) -> None:
    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=_FakeMps(mps)),
        cuda=_FakeCuda(cuda, cuda_count),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_select_device_accepts_explicit_cpu() -> None:
    assert select_device("cpu") == "cpu"


def test_select_device_accepts_cuda_when_available(monkeypatch) -> None:
    install_fake_torch(monkeypatch, cuda=True, mps=False, cuda_count=2)

    assert select_device("cuda") == "cuda"
    assert select_device("cuda:1") == "cuda:1"


def test_select_device_rejects_unavailable_cuda(monkeypatch) -> None:
    install_fake_torch(monkeypatch, cuda=False, mps=False)

    with pytest.raises(ValueError, match="CUDA requested"):
        select_device("cuda")


def test_select_device_auto_prefers_cuda_when_mps_is_unavailable(monkeypatch) -> None:
    install_fake_torch(monkeypatch, cuda=True, mps=False, cuda_count=1)

    assert select_device("auto") == "cuda"
