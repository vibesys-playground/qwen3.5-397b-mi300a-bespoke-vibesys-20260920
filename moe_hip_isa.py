"""Offline gfx942 ISA report for `moe_hip.hip`: no GPU needed, only a ROCm clang.

    python moe_hip_isa.py --rocm /opt/rocm            # or a TheRock wheel's _rocm_sdk_core
    python moe_hip_isa.py --rocm ... -D MOE_DEPTH=2   # a knob variant

Per kernel: VGPR/SGPR/LDS/scratch, waves per SIMD, and for the steady-state tile loop (the
innermost loop) the loads issued per trip, the `s_waitcnt vmcnt(N)` in front of each tile's
decode (N / 5 = tiles still in flight while one decodes), and VALU/MFMA counts per tile.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compile_asm(rocm: Path, defines: list[str], out: Path) -> str:
    clang = rocm / "lib" / "llvm" / "bin" / "clang++"
    if not clang.exists():
        clang = rocm / "llvm" / "bin" / "clang++"
    bitcode = next(
        p for p in (rocm / "lib/llvm/amdgcn/bitcode", rocm / "amdgcn/bitcode") if p.exists()
    )
    cmd = [
        str(clang), "-x", "hip", "--offload-arch=gfx942", "--cuda-device-only", "-O3", "-S",
        f"--rocm-path={rocm}", f"--rocm-device-lib-path={bitcode}", f"-I{rocm / 'include'}",
        *(f"-D{d}" for d in defines), "-o", str(out), str(HERE / "moe_hip.hip"),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)
    return out.read_text()


def kernel_meta(asm: str) -> dict[str, dict[str, int]]:
    """amdhsa.kernels metadata: name -> {vgpr_count, sgpr_count, group/private segment}."""
    meta = asm.split("amdhsa.kernels:", 1)[1]
    out = {}
    for entry in re.split(r"^  - ", meta, flags=re.M)[1:]:
        # top-level keys only (4-space indent); the nested kernel-argument list repeats .name
        fields = dict(re.findall(r"^    \.(\w+):\s+(\S+)", "    " + entry, flags=re.M))
        if "vgpr_count" not in fields:  # amdhsa.version's list items
            continue
        out[fields["name"]] = {
            k: int(fields[k])
            for k in (
                "vgpr_count",
                "sgpr_count",
                "group_segment_fixed_size",
                "private_segment_fixed_size",
            )
        }
    return out


def report(asm: str) -> None:
    meta = kernel_meta(asm)
    for name in re.findall(r"^(_Z\w*moe_kernel\w*):", asm, flags=re.M):
        m = meta[name]
        vgpr, sgpr = m["vgpr_count"], m["sgpr_count"]
        lds, scratch = m["group_segment_fixed_size"], m["private_segment_fixed_size"]
        waves = min(8, 512 // (-(-vgpr // 8) * 8))
        body = re.split(rf"^{name}:.*\n", asm, maxsplit=1, flags=re.M)[1].split("s_endpgm", 1)[0]
        # the innermost loop: blocks whose label comment names its header at the max depth
        depth = max(int(d) for d in re.findall(r"Depth=(\d+)", body))
        lines = body.splitlines()
        at = next(i for i, ln in enumerate(lines) if re.search(rf"Loop Header: Depth={depth}", ln))
        hdr = next(
            re.match(r"^\.(LBB\w+):", lines[i]).group(1)
            for i in range(at, -1, -1)
            if re.match(r"^\.LBB\w+:", lines[i])
        )
        loads = dx4 = mfma = valu = 0
        waits, pending_wait, inside = [], None, False
        for ln in lines:
            s = ln.strip()
            if re.match(r"^\.LBB\w+:", s):
                inside = s.startswith(f".{hdr}:") or f"Header={hdr[1:]}" in s
                continue
            if not inside or not s or s.startswith((";", ".")):
                continue
            op = s.split()[0]
            if op.startswith("global_load_dwordx4"):
                dx4 += 1
            elif op.startswith("global_load"):
                loads += 1
            elif op.startswith("s_waitcnt") and "vmcnt" in s:
                pending_wait = int(re.search(r"vmcnt\((\d+)\)", s).group(1))
            elif op.startswith("v_mfma"):
                if pending_wait is not None:
                    waits.append(pending_wait)
                pending_wait = None
                mfma += 1
            elif op.startswith("v_"):
                valu += 1
        tiles = max(1, mfma // 32)
        print(
            f"{name}\n  vgpr {vgpr} sgpr {sgpr} lds {lds} B scratch {scratch} B -> {waves} waves/SIMD"
        )
        print(
            f"  loop: {tiles} tiles/trip, {dx4 / tiles:.0f} x dwordx4 + {loads / tiles:.0f} x dword per "
            f"tile, {valu / tiles:.0f} VALU + {mfma / tiles:.0f} MFMA per tile (4 KB fp4), "
            f"vmcnt before decode: {waits}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocm", type=Path, required=True)
    ap.add_argument("-D", dest="defines", action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("moe_hip_gfx942.s"))
    args = ap.parse_args()
    report(compile_asm(args.rocm, args.defines, args.out))


if __name__ == "__main__":
    sys.exit(main())
