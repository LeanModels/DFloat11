"""A/B benchmark for DFloat11 decode kernels.

    python tests/bench_kernel.py --kernels dfloat11/decode.ptx:v1 dfloat11/decode.cu:production

The first kernel is the reference. Kernels are timed in interleaved rounds
(A B A B ...) so drift in clocks or temperature hits all of them equally, and
each kernel must decode a case correctly before it is timed on it.

Times are GPU-side (CUDA events, launches queued behind a GPU sleep so
Python launch overhead is excluded), and small cases rotate through enough
copies that every launch reads its inputs from DRAM rather than L2, as a
decode during inference would. Throughput is reported as GB/s of bfloat16
output. A workload matrix is used
rather than a single tensor so that a change which only helps one size or one
code-length profile is visible as such.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import cupy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

M = 1_000_000


def case_builders(full):
    """name -> builder. Sizes span 0.5 MB..1 GB of output; code-length
    profiles span ~1 to ~8 bits per exponent."""
    c = {}
    # Size sweep at the entropy of real weights (~2.6 bits/exponent).
    for n in (M // 20, M // 4, M, 4 * M, 64 * M):
        c[f'gauss_{n // 1000}k'] = lambda n=n: H.make_case('', H.gaussian_bf16(n + 1, 45))
    # Entropy sweep at a fixed 64M elements.
    c['dominant99_64M'] = lambda: H.make_case('', H.sampled_bf16(64 * M + 1, H.pmf_dominant(0.99), 51))
    c['dominant80_64M'] = lambda: H.make_case('', H.sampled_bf16(64 * M + 1, H.pmf_dominant(0.80), 52))
    c['geometric_64M'] = lambda: H.make_case('', H.sampled_bf16(64 * M + 1, H.pmf_geometric(40, 0.55), 53))
    c['uniform32_64M'] = lambda: H.make_case('', H.sampled_bf16(64 * M + 1, H.pmf_uniform(32), 54))
    c['uniform240_64M'] = lambda: H.make_case('', H.sampled_bf16(64 * M + 1, H.pmf_uniform(240), 55))
    if H.find_real_model():
        c['llama_q_proj0'] = lambda: H.make_case('', H.real_tensor(H.llama_block_names(0)[:1]))
        c['llama_block0'] = lambda: H.make_case('', H.real_tensor(H.llama_block_names(0)))
        c['llama_block15'] = lambda: H.make_case('', H.real_tensor(H.llama_block_names(15)))
        if full:
            c['llama_block31'] = lambda: H.make_case('', H.real_tensor(H.llama_block_names(31)))
            c['llama_embed'] = lambda: H.make_case('', H.real_tensor(['model.embed_tokens.weight']))
    if full:
        c['gauss_500M'] = lambda: H.make_case('', H.gaussian_bf16(500 * M + 3, 46))
    return c


COLD_BYTES = 64 << 20      # rotate through copies until the working set exceeds this
MAX_COPIES = 256
MAX_QUEUED = 400           # stay well below the driver's launch-queue depth


def case_bytes(case):
    return case.codes.size + case.sign_mantissa.size + 2 * case.n_elements + case.gaps.size


class Timer:
    """GPU-side timing of back-to-back launches.

    A GPU sleep is queued first so the CPU can enqueue every launch before the
    first one starts: the measured interval is then kernel time, not Python
    launch overhead, even for tensors that decode in a few microseconds.
    """

    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.clock_hz = cp.cuda.Device().attributes['ClockRate'] * 1e3
        self.enqueue_s = 50e-6

    def calibrate(self, fn, n=200):
        torch.cuda.synchronize()
        torch.cuda._sleep(int(self.clock_hz * 0.2))         # keep the GPU busy so enqueue does not block
        t0 = time.perf_counter()
        for i in range(n):
            fn(i)
        self.enqueue_s = (time.perf_counter() - t0) / n
        torch.cuda.synchronize()

    def __call__(self, fn, reps):
        torch.cuda.synchronize()
        torch.cuda._sleep(int(self.clock_hz * (reps * self.enqueue_s * 2 + 1e-3)))
        self.start.record()
        for i in range(reps):
            fn(i)
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) / 1e3 / reps


def bench_case(case, kernels, samples, min_sample_s, timer):
    copies = int(min(MAX_COPIES, max(1, np.ceil(COLD_BYTES / case_bytes(case)))))
    gs = [H.upload(case) for _ in range(copies)]
    outs = [torch.empty(case.n_elements, dtype=torch.int16, device='cuda:0') for _ in range(copies)]
    live = []
    for k in kernels:
        states = [k.prepare(g) for g in gs]
        ok, msg = H.run_checked(k, gs[0], states[0])
        if not ok:
            print(f'    {k.name}: not timed ({msg})')
            live.append(None)
        else:
            live.append(states)
    torch.cuda.empty_cache()

    def launcher(i):
        k, states = kernels[i], live[i]
        return lambda r: k.launch(gs[r % copies], states[r % copies], outs[r % copies].data_ptr())

    times = [[] for _ in kernels]
    reps = [1] * len(kernels)
    for i in range(len(kernels)):
        if live[i] is None:
            continue
        timer.calibrate(launcher(i))
        for _ in range(3):
            timer(launcher(i), 4)       # warm up
        t = timer(launcher(i), 8)
        reps[i] = int(min(MAX_QUEUED, max(copies, np.ceil(min_sample_s / t))))
    order = list(range(len(kernels)))
    for r in range(samples):
        for i in order[r % len(order):] + order[:r % len(order)]:   # rotate who goes first
            if live[i] is not None:
                times[i].append(timer(launcher(i), reps[i]))
    del gs, outs
    torch.cuda.empty_cache()
    return [np.array(t) if t else None for t in times], copies


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kernels', nargs='+', required=True, help='kernel.cu|.ptx[:abi]; the first is the reference')
    ap.add_argument('--abi-module', help='python file that registers extra ABIs (experiments)')
    ap.add_argument('--cases', nargs='*', help='substring filters on case names')
    ap.add_argument('--full', action='store_true', help='add the 1 GB tensors')
    ap.add_argument('--samples', type=int, default=15)
    ap.add_argument('--min-sample-ms', type=float, default=40.0)
    ap.add_argument('--json', help='write raw timings here')
    args = ap.parse_args()
    if args.abi_module:
        import importlib.util
        spec = importlib.util.spec_from_file_location('abi_module', args.abi_module)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    kernels = [H.Kernel(s) for s in args.kernels]
    timer = Timer()
    builders = case_builders(args.full)
    names = [n for n in builders if not args.cases or any(f in n for f in args.cases)]
    print(f'{torch.cuda.get_device_name(0)}; reference = {kernels[0].name}; '
          f'{args.samples} interleaved samples of >= {args.min_sample_ms:.0f} ms (GPU time; inputs rotated '
          f'through >= {COLD_BYTES >> 20} MB so small cases read from DRAM)')
    head = f'{"case":<17}{"elements":>12}{"bits":>6}'
    for i, k in enumerate(kernels):
        head += f'{("[%d] GB/s" % i):>11}{"spread":>8}' + ('' if i == 0 else f'{"vs[0]":>8}')
    print(head)
    for i, k in enumerate(kernels):
        print(f'  [{i}] {k.name}')

    raw, ratios = {}, [[] for _ in kernels]
    for name in names:
        case = H.cached_case('bench__' + name, builders[name])
        times, copies = bench_case(case, kernels, args.samples, args.min_sample_ms / 1e3, timer)
        out_gb = 2 * case.n_elements / 1e9
        row = f'{name:<17}{case.n_elements:>12,}{case.bits_per_symbol:>6.2f}'
        raw[name] = dict(n_elements=case.n_elements, bits=case.bits_per_symbol,
                         times={k.name: (t.tolist() if t is not None else None) for k, t in zip(kernels, times)})
        for i, t in enumerate(times):
            if t is None:
                row += f'{"-":>11}{"-":>8}' + ('' if i == 0 else f'{"-":>8}')
                continue
            med = np.median(t)
            spread = (np.percentile(t, 75) - np.percentile(t, 25)) / med
            row += f'{out_gb / med:>11.2f}{spread:>7.1%} '
            if i > 0:
                if times[0] is None:
                    row += f'{"-":>8}'
                else:
                    ratio = np.median(times[0]) / med
                    ratios[i].append(ratio)
                    row += f'{ratio:>7.2f}x'
        print(row, flush=True)

    for i in range(1, len(kernels)):
        if ratios[i]:
            r = np.array(ratios[i])
            print(f'[{i}] vs [0]: geomean {np.exp(np.log(r).mean()):.3f}x, min {r.min():.3f}x, max {r.max():.3f}x '
                  f'over {r.size} cases')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(raw, f)


if __name__ == '__main__':
    main()
