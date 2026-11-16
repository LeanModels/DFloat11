"""Bit-exact correctness check for a DFloat11 decode kernel.

    python tests/verify_kernel.py                              # production kernel
    python tests/verify_kernel.py --kernel path/to/k.cu:abi    # a candidate
    python tests/verify_kernel.py --full                       # + real weights, 500M-element tensors
    compute-sanitizer --tool memcheck  python tests/verify_kernel.py --tiny
    compute-sanitizer --tool racecheck python tests/verify_kernel.py --tiny

Every case is "decode(encode(x)) must equal x": the expected output is the
original tensor, never the output of another kernel.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

DEFAULT_KERNEL = os.path.join(H.REPO, 'dfloat11', 'decode.cu') + ':production'


class Report:
    def __init__(self):
        self.passed, self.failed, self.skipped = 0, [], []

    def check(self, name, ok, msg=''):
        if ok:
            self.passed += 1
        else:
            self.failed.append((name, msg))
            print(f'  FAIL {name}: {msg}')

    def skip(self, name, why):
        self.skipped.append((name, why))


# --------------------------------------------------------------------------
# 1. The test encoder must agree with the repo's encoder.
# --------------------------------------------------------------------------

def small_tensors():
    """(name, uint16 tensor) pairs covering every code-length profile."""
    rng = np.random.default_rng(0)
    yield 'gauss', H.gaussian_bf16(30011, 1)
    yield 'gauss_wide', H.gaussian_bf16(30011, 2, std=3.0)
    yield 'zeros', np.zeros(5000, np.uint16)
    yield 'neg_zeros_mix', rng.choice(np.array([0, 0x8000], np.uint16), 5000)
    yield 'constant', np.full(5000, 0x3F80, np.uint16)
    yield 'two_symbols', H.sampled_bf16(20000, np.array([0] * 100 + [1, 1]), 3)
    yield 'dominant90', H.sampled_bf16(40000, H.pmf_dominant(0.9), 4)
    yield 'dominant99', H.sampled_bf16(40000, H.pmf_dominant(0.99), 5)
    yield 'uniform16', H.sampled_bf16(30000, H.pmf_uniform(16), 6)
    yield 'uniform32', H.sampled_bf16(30000, H.pmf_uniform(32), 7)
    yield 'uniform200', H.sampled_bf16(40000, H.pmf_uniform(200), 8)
    yield 'uniform240', H.sampled_bf16(40000, H.pmf_uniform(240), 9)
    yield 'denormals', H.sampled_bf16(20000, np.array([5, 3, 1, 1]), 10)
    for k, ratio in ((12, 0.5), (20, 0.5), (16, 0.62)):
        yield f'geometric{k}_{ratio}', H.sampled_bf16(60000, H.pmf_geometric(k, ratio), 11)


def fibonacci_tensor(k, seed=0):
    """Exponent counts 1,1,2,3,5,...: the Huffman tree is a comb, so code
    lengths reach k-1 bits and every lookup-table level gets exercised."""
    counts = [1, 1]
    while len(counts) < k:
        counts.append(counts[-1] + counts[-2])
    rng = np.random.default_rng(seed)
    exp = np.repeat(np.arange(k, dtype=np.uint8) + 100, counts)
    rng.shuffle(exp)
    return H.bf16_from_parts(exp, rng)


def check_encoder(rep):
    print('[encoder] fast test encoder vs dfloat11_utils.encode')
    tensors = list(small_tensors()) + [(f'fib{k}', fibonacci_tensor(k)) for k in (10, 18, 22)]
    for n in (1, 2, 3, 7, 8, 9, 63, 64, 65, 1000):
        tensors.append((f'gauss_n{n}', H.gaussian_bf16(n, 100 + n)))
    for name, w in tensors:
        exp, _ = H.split_bf16(w)
        try:
            codec = H.build_codec(exp)
        except H.UnsupportedCodec as e:
            rep.skip(f'encoder/{name}', str(e))
            continue
        fast = H.fast_encode(exp, codec, chunk=7919)   # small chunks: exercise chunk seams
        ref = H.reference_encode(exp, codec)
        same = all(np.array_equal(a, b) for a, b in zip(fast, ref))
        rep.check(f'encoder/{name}', same, 'fast_encode differs from reference encoder')


# --------------------------------------------------------------------------
# 2. Kernel cases
# --------------------------------------------------------------------------

def run_case(rep, kernel, case, repeats=1, **upload_kw):
    g = H.upload(case, **upload_kw)
    inputs_before = {k: t.clone() for k, t in g.tensors.items()} if case.n_elements < (1 << 23) else None
    state = kernel.prepare(g)
    for r in range(repeats):
        ok, msg = H.run_checked(kernel, g, state)
        rep.check(case.name + (f'#{r}' if repeats > 1 else ''), ok, msg)
        if not ok:
            break
    if inputs_before is not None:
        same = all(torch.equal(t, inputs_before[k]) for k, t in g.tensors.items())
        rep.check(case.name + '/inputs-unmodified', same, 'kernel modified an input buffer')
    del g
    return state


def try_case(rep, name, w, **meta):
    try:
        return H.make_case(name, w, **meta)
    except H.UnsupportedCodec as e:
        rep.skip(name, str(e))
        return None


def boundary_sweep(rep, kernel, name, w, targets):
    """Trim the tensor so the encoded stream has exactly `target` bytes, both
    with a partial final byte (EOF padding) and, where possible, without."""
    exp, _ = H.split_bf16(w)
    codec = H.build_codec(exp)
    cum = np.cumsum(codec.lens[exp].astype(np.int64))
    nbytes = (cum + 7) // 8
    for target in targets:
        idx = np.nonzero(nbytes == target)[0]
        if idx.size == 0:
            continue
        picks = {int(idx[-1]) + 1, int(idx[0]) + 1}
        exact = idx[cum[idx] % 8 == 0]
        if exact.size:
            picks.add(int(exact[0]) + 1)
        for n in sorted(picks):
            case = H.make_case(f'{name}/bytes={target}/n={n}', w[:n], codec)
            assert case.n_bytes == target
            run_case(rep, kernel, case)


def check_kernel(rep, kernel, level):
    tiny = level == 'tiny'

    print('[kernel] element-count sweep')
    for dist, gen in (('gauss', lambda n, s: H.gaussian_bf16(n, s)),
                      ('dominant', lambda n, s: H.sampled_bf16(n, H.pmf_dominant(0.95), s)),
                      ('uniform240', lambda n, s: H.sampled_bf16(n, H.pmf_uniform(240), s))):
        for n in ([1, 2, 9, 65] if tiny else list(range(1, 70)) + [100, 127, 128, 129, 511, 512, 513, 1000]):
            case = try_case(rep, f'count/{dist}/n={n}', gen(n, n))
            if case:
                run_case(rep, kernel, case)

    print('[kernel] stream-length boundaries (thread window, lookahead, block)')
    b = H.BLOCK_BYTES
    targets = list(range(1, 27)) + [x + d for x in (b, 2 * b, 3 * b) for d in range(-9, 10)]
    if tiny:
        targets = [1, 7, 8, 9, 12, 13, b - 1, b, b + 1, 2 * b]
    for name, w in (('gauss', H.gaussian_bf16(60000, 21)),
                    ('dominant', H.sampled_bf16(120000, H.pmf_dominant(0.95), 22)),
                    ('uniform240', H.sampled_bf16(20000, H.pmf_uniform(240), 23)),
                    ('fib24', fibonacci_tensor(24, 1))):
        boundary_sweep(rep, kernel, f'boundary/{name}', w, targets)

    print('[kernel] code-length profiles')
    for name, w in small_tensors():
        case = try_case(rep, f'profile/{name}', w)
        if case:
            run_case(rep, kernel, case, repeats=2)
    for k in ((12, 26) if tiny else (10, 12, 17, 18, 20, 24, 25, 26, 28, 30, 32, 33, 34, 36)):
        case = try_case(rep, f'profile/fib{k}', fibonacci_tensor(k, k))
        if case:
            run_case(rep, kernel, case)
            print(f'    fib{k}: {case.n_elements} elements, max code {case.meta["max_code_len"]} bits, '
                  f'{case.meta["n_luts"]} luts')

    print('[kernel] random exponent distributions')
    rng = np.random.default_rng(1234)
    for i in range(4 if tiny else 60):
        k = int(rng.integers(2, 240))
        kind = i % 4
        if kind == 0:
            pmf = rng.dirichlet(np.full(k, 0.05))          # very spiky
        elif kind == 1:
            pmf = rng.dirichlet(np.full(k, 1.0))
        elif kind == 2:
            pmf = 1.0 / np.arange(1, k + 1) ** rng.uniform(0.5, 3.0)   # zipf
            rng.shuffle(pmf)
        else:
            pmf = rng.uniform(0.3, 0.8) ** np.arange(k)    # geometric, long codes
            rng.shuffle(pmf)
        full = np.zeros(240)
        full[rng.choice(240, k, replace=False)] = pmf + 1e-300
        n = int(rng.integers(1, 20000 if tiny else 400000))
        case = try_case(rep, f'random/{i}/k={k}/n={n}', H.sampled_bf16(n, full, 5000 + i))
        if case:
            run_case(rep, kernel, case)

    print('[kernel] misaligned input pointers')
    w = H.gaussian_bf16(100003 if tiny else 1000003, 31)
    mis = H.make_case('misaligned', w)
    for off in range(1, 8):
        mis.name = f'misaligned/codes+{off}'
        run_case(rep, kernel, mis, codes_offset=off)
    for kw in (dict(sm_offset=1), dict(sm_offset=3), dict(luts_offset=1), dict(gaps_offset=1),
               dict(codes_offset=4, sm_offset=5, luts_offset=3, gaps_offset=7)):
        mis.name = 'misaligned/' + ','.join(f'{k}={v}' for k, v in kw.items())
        run_case(rep, kernel, mis, **kw)
    w = fibonacci_tensor(26, 3)
    mis = H.make_case('misaligned/fib26', w)
    for off in (1, 2, 3, 4):
        mis.name = f'misaligned/fib26/codes+{off}'
        run_case(rep, kernel, mis, codes_offset=off)

    if tiny:
        return

    print('[kernel] medium and large tensors, repeated (races show up as flaky output)')
    big = [
        ('large/gauss_4M', lambda: H.make_case('', H.gaussian_bf16(4_000_037, 41))),
        ('large/dominant99_8M', lambda: H.make_case('', H.sampled_bf16(8_000_011, H.pmf_dominant(0.99), 42))),
        ('large/uniform240_4M', lambda: H.make_case('', H.sampled_bf16(4_000_003, H.pmf_uniform(240), 43))),
        ('large/geometric_8M', lambda: H.make_case('', H.sampled_bf16(8_000_009, H.pmf_geometric(40, 0.55), 44))),
        ('large/gauss_64M', lambda: H.make_case('', H.gaussian_bf16(64_000_001, 45))),
    ]
    if level == 'full':
        big.append(('large/gauss_500M', lambda: H.make_case('', H.gaussian_bf16(500_000_003, 46))))
        if H.find_real_model():
            for i in (0, 15, 31):
                big.append((f'real/llama_block{i}', lambda i=i: H.make_case('', H.real_tensor(H.llama_block_names(i)))))
            big.append(('real/llama_q_proj0', lambda: H.make_case('', H.real_tensor(H.llama_block_names(0)[:1]))))
            big.append(('real/llama_embed', lambda: H.make_case('', H.real_tensor(['model.embed_tokens.weight']))))
        else:
            rep.skip('real/*', 'no local bfloat16 checkpoint (set DF11_REAL_MODEL)')
    for name, builder in big:
        try:
            case = H.cached_case(name.replace('/', '__'), builder)
        except H.UnsupportedCodec as e:
            rep.skip(name, str(e))
            continue
        case.name = name
        t0 = time.time()
        run_case(rep, kernel, case, repeats=5)
        print(f'    {name}: {case.n_elements:,} elements, {case.bits_per_symbol:.2f} bits/exp, '
              f'{time.time() - t0:.1f}s')
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--kernel', default=DEFAULT_KERNEL, help='kernel.cu|.ptx[:abi]')
    ap.add_argument('--abi-module', help='python file that registers extra ABIs (experiments)')
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--tiny', action='store_true', help='small subset, for compute-sanitizer')
    g.add_argument('--full', action='store_true', help='add real weights and 500M-element tensors')
    args = ap.parse_args()
    if args.abi_module:
        import importlib.util
        spec = importlib.util.spec_from_file_location('abi_module', args.abi_module)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    level = 'tiny' if args.tiny else 'full' if args.full else 'default'
    rep = Report()
    kernel = H.Kernel(args.kernel)
    print(f'kernel: {kernel.name}  ({torch.cuda.get_device_name(0)})')
    t0 = time.time()
    check_encoder(rep)
    check_kernel(rep, kernel, level)

    print(f'\n{rep.passed} checks passed, {len(rep.failed)} failed, {len(rep.skipped)} skipped '
          f'in {time.time() - t0:.0f}s')
    for name, why in rep.skipped:
        print(f'  skipped {name}: {why}')
    for name, msg in rep.failed[:40]:
        print(f'  FAILED  {name}: {msg}')
    sys.exit(1 if rep.failed else 0)


if __name__ == '__main__':
    main()
