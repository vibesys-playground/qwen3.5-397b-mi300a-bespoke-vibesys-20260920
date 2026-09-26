import json, sys, collections, statistics
D, tag = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""
def kcat(seg, n):
    l = n.lower()
    if "rccl" in l: return "all-reduce"
    if seg == "moe":
        if "moe_kernel" in l: return "MoE expert GEMMs"
        if "_bw_prep" in l or "_bw_combine" in l: return "MoE prep+combine"
        if "topk" in l or "bitonic" in l or "reduce_kernel" in l or "softmax" in l: return "MoE routing (softmax/topk/renorm)"
        if "cijk" in l or "gemv" in l: return "MoE dense GEMMs (router, shared expert)"
        if "rmsnorm" in l: return "norms"
        return "MoE glue (sigmoid/mul/add/cast)"
    if seg == "dn":
        if "_delta_rule" in l or "gated_rmsnorm" in l or "conv" in l or "transpose" in l: return "DeltaNet kernels"
        if "cijk" in l or "gemv" in l: return "mixer dense GEMMs (proj)"
        if "rmsnorm" in l: return "norms"
        return "DeltaNet state copies/glue"
    if seg == "attn":
        if "_splitk" in l or "_combine" in l: return "attention kernels"
        if "cijk" in l or "gemv" in l: return "mixer dense GEMMs (proj)"
        if "rmsnorm" in l: return "norms"
        return "attention glue (rope/kv write/gate)"
    if "cijk" in l or "gemv" in l: return "lm_head/embed/other GEMM"
    return "step glue (outside layers)"
for f in sorted(__import__("glob").glob(f"{D}/trace_{tag}r0_b*.json"), key=lambda s: int(s.split("_b")[-1][:-5])):
    ev = json.load(open(f))["traceEvents"]
    ks = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"], key=lambda e: e["ts"])
    ar = [i for i, e in enumerate(ks) if "rccl" in e["name"]]
    tot = collections.defaultdict(float); nsteps = len(ar) / 120
    for j in range(len(ar)):
        s = ar[j]; t = ar[j + 1] if j + 1 < len(ar) else len(ks)
        names = [e["name"] for e in ks[s + 1:t]]
        seg = "moe" if any("moe_kernel" in n for n in names) else "dn" if any("_delta_rule" in n for n in names) else "attn" if any("_splitk" in n for n in names) else "other"
        for e in ks[s:t]:
            tot[kcat(seg, e["name"])] += e["dur"]
    ard = sorted(e["dur"] for e in ks if "rccl" in e["name"])
    print(f"\n{f.split('/')[-1]}: steps~{nsteps:.2f} kernels/step={len(ks)/nsteps:.0f} total={sum(tot.values())/nsteps/1e3:.2f} ms  AR us p10={ard[len(ard)//10]:.0f} p50={statistics.median(ard):.0f} p90={ard[9*len(ard)//10]:.0f}")
    for k, v in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"  {k:42s} {v/nsteps/1e3:6.2f}")
