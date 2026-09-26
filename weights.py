"""Thread-safe safetensors access over a directory of shards.

Reads use `os.preadv` (releases the GIL, no shared file offset), so many threads can pull
tensors at once. That matters on the APU, where one sequential reader gets ~45 MB/s.
"""

import json
import os
import struct
from pathlib import Path

import torch

LOAD_FADVISE = os.environ.get("SEED_LOAD_FADVISE", "0") == "1"
"""Drop each tensor's file pages from the page cache once read. On MI300A the page cache
shares each NUMA node's memory with that node's GPU allocations; see host_numa."""

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "BOOL": torch.bool,
}


class Checkpoint:
    """Maps tensor name -> (shard, dtype, shape, byte range) from shard headers; bytes load on demand."""

    def __init__(self, path: str | Path) -> None:
        self.files = sorted(Path(path).glob("*.safetensors"))
        if not self.files:
            raise FileNotFoundError(f"no *.safetensors files under {path}")
        self._fds: dict[Path, int] = {}
        self._meta: dict[str, tuple[Path, torch.dtype, tuple[int, ...], int, int]] = {}
        for f in self.files:
            fd = os.open(f, os.O_RDONLY)
            self._fds[f] = fd
            (hlen,) = struct.unpack("<Q", os.pread(fd, 8, 0))
            header = json.loads(os.pread(fd, hlen, 8))
            base = 8 + hlen
            for name, info in header.items():
                if name == "__metadata__":
                    continue
                lo, hi = info["data_offsets"]
                self._meta[name] = (
                    f,
                    _DTYPES[info["dtype"]],
                    tuple(info["shape"]),
                    base + lo,
                    hi - lo,
                )

    def has(self, name: str) -> bool:
        return name in self._meta

    def load(
        self, name: str, device: str | torch.device, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """Read one tensor, move it to `device`, cast to `dtype` unless None. Safe from many threads."""
        f, src_dtype, shape, offset, size = self._meta[name]
        buf = bytearray(size)
        view, got = memoryview(buf), 0
        while got < size:
            n = os.preadv(self._fds[f], [view[got:]], offset + got)
            if n <= 0:
                raise OSError(f"short read for {name}")
            got += n
        if LOAD_FADVISE:
            os.posix_fadvise(self._fds[f], offset, size, os.POSIX_FADV_DONTNEED)
        tensor = (
            torch.frombuffer(buf, dtype=src_dtype).reshape(shape)
            if size
            else torch.empty(shape, dtype=src_dtype)
        )
        return tensor.to(device=device, dtype=dtype or src_dtype)
