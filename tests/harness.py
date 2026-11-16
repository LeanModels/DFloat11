"""Shared machinery for verifying and benchmarking the DFloat11 decode kernel.

Nothing in here depends on the kernel under test being correct:

* test data is produced by ``fast_encode`` (a vectorised re-implementation of
  ``dfloat11_utils.encode`` that is cross-checked against the original on small
  inputs by ``verify_kernel.py``), and
* the expected output of every case is the original bfloat16 tensor, so a
  kernel passes only if it reproduces the input bit for bit.
"""

import contextlib
import glob
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import cupy as cp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dfloat11.dfloat11_utils import (  # noqa: E402
    get_32bit_codec, get_luts, get_decode_tables, get_decode_shared_mem_size, encode as slow_encode,
)

THREADS_PER_BLOCK = 512
BYTES_PER_THREAD = 8
BLOCK_BYTES = THREADS_PER_BLOCK * BYTES_PER_THREAD
CACHE_DIR = os.environ.get('DF11_TEST_CACHE', os.path.join(REPO, 'tests', '.cache'))


# --------------------------------------------------------------------------
# Codec + encoder
# --------------------------------------------------------------------------

class UnsupportedCodec(Exception):
    """The repo's encoder cannot represent this code (pre-existing format limits)."""


@dataclass
class Codec:
    lens: np.ndarray        # [256] uint8, 0 for unused symbols
    codes: np.ndarray       # [256] uint32
    eof_len: int
    eof_code: int
    luts: np.ndarray        # [n_luts, 256] uint8, last row = code lengths
    hf_codec: object = None  # dahuffman codec, for the slow reference encoder


def build_luts(table) -> np.ndarray:
    """dfloat11_utils.get_luts(), except that entries below the first defined
    one are back-filled instead of raising UnboundLocalError."""
    prefixes = ['']
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, '0')[:((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)
    prefixes.sort(key=len)

    luts = np.zeros((len(prefixes) + 1, 256), dtype=np.uint8)
    cur = None      # like get_luts, a hole inherits the previous entry, even across tables
    for pi, p in enumerate(prefixes):
        entries, pl = {}, len(p) // 8
        for key, (bits, val) in table.items():
            if not isinstance(key, int):
                continue
            code = bin(val)[2:].rjust(bits, '0')
            if code.startswith(p):
                if (bits - 1) // 8 == pl:
                    k, v = int(code[pl * 8:].ljust(8, '0'), 2), key
                else:
                    k, v = int(code[pl * 8:pl * 8 + 8], 2), 256 - prefixes.index(code[:pl * 8 + 8])
                assert entries.get(k, v) == v
                entries[k] = v
        for i in range(256):
            cur = entries.get(i, cur)
            luts[pi, i] = entries[min(entries)] if cur is None else cur
    for key, (bits, _) in table.items():
        if isinstance(key, int):
            luts[-1, key] = bits
    return luts


def build_codec(exponents: np.ndarray) -> Codec:
    """Build the Huffman code exactly as compress_model() does."""
    hist = np.bincount(exponents, minlength=256)
    counter = {int(s): int(c) for s, c in enumerate(hist) if c > 0}
    if max(counter) >= 240:
        raise UnsupportedCodec('exponent symbols >= 240 collide with LUT pointers')
    with contextlib.redirect_stdout(io.StringIO()):
        hf_codec, _, table = get_32bit_codec(counter)
        try:
            luts = get_luts(table).numpy()
            assert np.array_equal(luts, build_luts(table)), 'harness build_luts drifted from get_luts'
        except UnboundLocalError:
            # get_luts() crashes when the all-zero code belongs to EOF (common
            # for tiny alphabets). The stream is still well defined, so build
            # the same tables with that hole filled.
            luts = build_luts(table)
        except ValueError as e:
            # A long code whose 8-bit prefix has no shorter sibling.
            raise UnsupportedCodec(f'get_luts cannot build tables ({e})')
    if luts.shape[0] - 1 > 17:
        raise UnsupportedCodec(f'{luts.shape[0] - 1} decode tables; pointers only address 16')

    lens = np.zeros(256, np.uint8)
    codes = np.zeros(256, np.uint32)
    eof_len = eof_code = None
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            lens[key], codes[key] = bits, val
        else:
            eof_len, eof_code = bits, val
    assert lens.max() <= 32 and eof_len is not None
    return Codec(lens, codes, eof_len, eof_code, luts, hf_codec)


def fast_encode(sym: np.ndarray, codec: Codec, chunk: int = 1 << 24):
    """Vectorised equivalent of dfloat11_utils.encode().

    Returns (encoded uint8, packed gaps uint8, output_positions uint32).
    """
    n = sym.size
    assert n > 0
    lens64 = codec.lens.astype(np.int64)
    codes64 = codec.codes.astype(np.uint64)

    total_bits = int((np.bincount(sym, minlength=256) * lens64).sum())
    size = total_bits % 8
    n_bytes = (total_bits + 7) // 8
    last_start = total_bits if size > 0 else total_bits - int(lens64[sym[-1]])
    n_win = (last_start >> 6) + 1
    n_blk = (last_start >> 15) + 1

    words = torch.zeros((n_bytes + 3) // 4 + 2, dtype=torch.int64)
    gaps = np.zeros(n_win, np.uint8)
    outpos = np.zeros(n_blk + 1, np.uint32)

    bitpos, prev_win, prev_blk = 0, -1, -1
    for a in range(0, n, chunk):
        s = sym[a:a + chunk]
        L = lens64[s]
        ends = np.cumsum(L) + bitpos
        starts = ends - L
        # Left-align each code inside a 64-bit lane that covers words idx, idx+1.
        v = codes64[s] << (64 - L - (starts & 31)).astype(np.uint64)
        idx = torch.from_numpy(starts >> 5)
        words.index_add_(0, idx, torch.from_numpy((v >> np.uint64(32)).astype(np.int64)))
        words.index_add_(0, idx + 1, torch.from_numpy((v & np.uint64(0xFFFFFFFF)).astype(np.int64)))

        win = starts >> 6
        first = np.empty(win.size, bool)
        first[0] = win[0] != prev_win
        first[1:] = win[1:] != win[:-1]
        gaps[win[first]] = starts[first] & 63

        blk = starts >> 15
        first[0] = blk[0] != prev_blk
        first[1:] = blk[1:] != blk[:-1]
        outpos[blk[first]] = a + np.nonzero(first)[0]

        bitpos, prev_win, prev_blk = int(ends[-1]), int(win[-1]), int(blk[-1])

    assert bitpos == total_bits
    if size > 0:
        # The reference encoder treats EOF as one more symbol start and emits
        # only as many of its leading bits as fit in the final byte.
        if (total_bits >> 6) != prev_win:
            gaps[total_bits >> 6] = total_bits & 63
        if (total_bits >> 15) != prev_blk:
            outpos[total_bits >> 15] = n
        room = 8 - size
        if codec.eof_len >= room:
            bits = codec.eof_code >> (codec.eof_len - room)
        else:
            bits = codec.eof_code << (room - codec.eof_len)
        byte_idx = total_bits >> 3
        words[byte_idx >> 2] += bits << ((3 - (byte_idx & 3)) * 8)
    outpos[-1] = n

    assert int(words.max()) < (1 << 32)
    encoded = words.numpy().astype('>u4').view(np.uint8)[:n_bytes].copy()

    blocks = -(-n_bytes // BLOCK_BYTES)
    all_gaps = np.zeros(THREADS_PER_BLOCK * blocks, np.uint8)
    all_gaps[:n_win] = gaps
    assert all_gaps.max() < 32
    gap_bits = ((all_gaps[:, None] >> np.arange(4, -1, -1, dtype=np.uint8)) & 1).astype(np.uint8)
    return encoded, np.packbits(gap_bits.ravel()), outpos


def reference_encode(sym: np.ndarray, codec: Codec):
    """The repo's own (slow, per-element) encoder."""
    with contextlib.redirect_stderr(io.StringIO()):
        enc, gaps, outpos = slow_encode(sym.tolist(), codec.hf_codec, BYTES_PER_THREAD, THREADS_PER_BLOCK)
    return enc, gaps, outpos


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

@dataclass
class Case:
    name: str
    expected: np.ndarray           # [n] uint16, the original bfloat16 bit patterns
    luts: np.ndarray               # [n_luts, 256] uint8
    codes: np.ndarray              # [n_bytes] uint8
    sign_mantissa: np.ndarray      # [n] uint8
    output_positions: np.ndarray   # [blocks + 1] uint32
    gaps: np.ndarray               # packed 5-bit gaps
    meta: dict = field(default_factory=dict)

    @property
    def n_elements(self):
        return self.expected.size

    @property
    def n_bytes(self):
        return self.codes.size

    @property
    def bits_per_symbol(self):
        return 8.0 * self.n_bytes / self.n_elements


def split_bf16(w: np.ndarray):
    """uint16 bfloat16 bit patterns -> (exponent byte, sign|mantissa byte)."""
    exp = ((w >> 7) & 0xFF).astype(np.uint8)
    rest = (((w >> 8) & 0x80) | (w & 0x7F)).astype(np.uint8)
    return exp, rest


def make_case(name: str, w: np.ndarray, codec: Codec = None, **meta) -> Case:
    w = np.ascontiguousarray(w, dtype=np.uint16)
    exp, rest = split_bf16(w)
    codec = codec or build_codec(exp)
    codes, gaps, outpos = fast_encode(exp, codec)
    meta = dict(meta, max_code_len=int(codec.lens.max()), min_code_len=int(codec.lens[codec.lens > 0].min()),
                n_luts=int(codec.luts.shape[0]))
    return Case(name, w, codec.luts, codes, rest, outpos, gaps, meta)


def cached_case(name: str, builder) -> Case:
    """Build a (large) case once and keep it on disk."""
    path = os.path.join(CACHE_DIR, 'cases', name + '.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        return Case(name, z['expected'], z['luts'], z['codes'], z['sign_mantissa'],
                    z['output_positions'], z['gaps'], json.loads(str(z['meta'])))
    case = builder()
    case.name = name
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, expected=case.expected, luts=case.luts, codes=case.codes,
             sign_mantissa=case.sign_mantissa, output_positions=case.output_positions,
             gaps=case.gaps, meta=json.dumps(case.meta))
    os.replace(tmp, path)
    return case


def bf16_from_parts(exp: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Attach random sign/mantissa bits to the given exponent bytes."""
    rest = rng.integers(0, 256, size=exp.size, dtype=np.uint8).astype(np.uint16)
    return ((rest & 0x80) << 8) | (exp.astype(np.uint16) << 7) | (rest & 0x7F)


def gaussian_bf16(n: int, seed: int, std: float = 0.02) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    out = np.empty(n, np.uint16)
    for a in range(0, n, 1 << 24):
        m = min(1 << 24, n - a)
        t = (torch.randn(m, generator=g) * std).to(torch.bfloat16)
        out[a:a + m] = t.view(torch.int16).numpy().view(np.uint16)
    return out


def sampled_bf16(n: int, pmf: np.ndarray, seed: int) -> np.ndarray:
    """bfloat16 values whose exponent byte follows `pmf` (length <= 240)."""
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(pmf / pmf.sum())
    out = np.empty(n, np.uint16)
    for a in range(0, n, 1 << 24):
        m = min(1 << 24, n - a)
        exp = np.minimum(np.searchsorted(cdf, rng.random(m)), len(pmf) - 1).astype(np.uint8)
        out[a:a + m] = bf16_from_parts(exp, rng)
    return out


# Exponent distributions, chosen to span the range of code-length profiles.
def pmf_dominant(p=0.9):
    """One symbol with a 1-bit code: the most symbols a thread can ever see."""
    pmf = np.zeros(140)
    pmf[120] = p
    pmf[110:120] = (1 - p) / 10
    return pmf


def pmf_uniform(k):
    pmf = np.zeros(240)
    pmf[:k] = 1.0
    return pmf


def pmf_geometric(k=40, ratio=0.5):
    """Geometric tail: code lengths grow by one per symbol, up to the 32-bit cap."""
    return ratio ** np.arange(k)


def find_real_model():
    env = os.environ.get('DF11_REAL_MODEL')
    if env:
        return env
    hits = glob.glob(os.path.expanduser(
        '~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/*/model.safetensors.index.json'))
    return os.path.dirname(hits[0]) if hits else None


def real_tensor(names) -> np.ndarray:
    """Concatenate bfloat16 tensors from a local HF checkpoint, as compress_model() would."""
    from safetensors import safe_open
    root = find_real_model()
    with open(os.path.join(root, 'model.safetensors.index.json')) as f:
        index = json.load(f)['weight_map']
    parts = []
    for name in names:
        with safe_open(os.path.join(root, index[name]), framework='pt') as f:
            t = f.get_tensor(name)
        assert t.dtype == torch.bfloat16, f'{name} is {t.dtype}'
        parts.append(t.flatten().view(torch.int16).numpy().view(np.uint16))
    return np.concatenate(parts)


LLAMA_BLOCK = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
               'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']


def llama_block_names(i):
    return [f'model.layers.{i}.{m}.weight' for m in LLAMA_BLOCK]


# --------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------

def find_nvcc():
    for cand in (os.environ.get('DF11_NVCC'),
                 os.path.expanduser('~/miniconda3/envs/nvcc122/bin/nvcc'),
                 shutil.which('nvcc'), '/usr/local/cuda/bin/nvcc'):
        if cand and os.path.exists(cand):
            return cand
    raise RuntimeError('nvcc not found; set DF11_NVCC')


def compile_ptx(src: str) -> str:
    """Compile a .cu file to PTX (cached by content); .ptx files pass through."""
    if src.endswith('.ptx'):
        return src
    nvcc = find_nvcc()
    with open(src, 'rb') as f:
        digest = hashlib.sha1(f.read() + nvcc.encode()).hexdigest()[:16]
    out = os.path.join(CACHE_DIR, 'ptx', f'{os.path.basename(src)[:-3]}-{digest}.ptx')
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run([nvcc, '-O3', '-ptx', src, '-o', out], check=True)
    return out


@dataclass
class GpuCase:
    """A Case uploaded to the GPU. Pointers are kept separately from the
    tensors so tests can substitute deliberately misaligned copies."""
    case: Case
    tensors: dict
    ptr: dict
    n_luts: int
    n_bytes: int
    n_elements: int
    max_block_elements: int
    expected: torch.Tensor

    @property
    def blocks(self):
        return -(-self.n_bytes // BLOCK_BYTES)


def upload(case: Case, device='cuda:0', codes_offset=0, sm_offset=0, luts_offset=0, gaps_offset=0) -> GpuCase:
    def put(arr, offset=0):
        t = torch.from_numpy(np.ascontiguousarray(arr).view(np.uint8).reshape(-1))
        if offset == 0:
            g = t.to(device)
            return g, g.data_ptr()
        g = torch.zeros(t.numel() + 16, dtype=torch.uint8, device=device)
        g[offset:offset + t.numel()] = t.to(device)
        return g, g.data_ptr() + offset

    tensors, ptr = {}, {}
    for key, arr, off in (('luts', case.luts, luts_offset), ('codes', case.codes, codes_offset),
                          ('sign_mantissa', case.sign_mantissa, sm_offset),
                          ('output_positions', case.output_positions, 0), ('gaps', case.gaps, gaps_offset)):
        tensors[key], ptr[key] = put(arr, off)
    op = case.output_positions.astype(np.int64)
    expected = torch.from_numpy(case.expected.view(np.int16)).to(device)
    return GpuCase(case, tensors, ptr, case.luts.shape[0], case.n_bytes, case.n_elements,
                   int((op[1:] - op[:-1]).max()), expected)


class Kernel:
    """A compiled decode kernel plus the host-side launch convention ("abi")."""

    def __init__(self, spec: str):
        # spec = "path/to/kernel.cu[:abi]"
        src, _, abi = spec.partition(':')
        self.src, self.abi = src, abi or 'v1'
        self.name = os.path.basename(src) + (f':{abi}' if abi else '')
        self.module = cp.RawModule(path=compile_ptx(src))
        self.fn = self.module.get_function('decode')
        self._prepare, self._launch = ABIS[self.abi]

    def prepare(self, g: GpuCase):
        """Host-side, once-per-tensor work (what the loader would do)."""
        return self._prepare(self, g)

    def launch(self, g: GpuCase, state, out_ptr: int):
        self._launch(self, g, state, out_ptr)


def _prepare_v1(k, g):
    return THREADS_PER_BLOCK * 4 + 4 + g.max_block_elements * 2


def _launch_v1(k, g, shared_mem, out_ptr):
    k.fn((g.blocks,), (THREADS_PER_BLOCK,), (
        g.ptr['luts'], g.ptr['codes'], g.ptr['sign_mantissa'], g.ptr['output_positions'], g.ptr['gaps'],
        out_ptr, g.n_luts, g.n_bytes, g.n_elements), shared_mem=shared_mem)


def _prepare_production(k, g):
    """Exactly what dfloat11.py does at load time."""
    tables = get_decode_tables(torch.from_numpy(g.case.luts)).to(g.expected.device)
    return dict(tables=tables, shared_mem=get_decode_shared_mem_size(g.case.output_positions))


def _launch_production(k, g, st, out_ptr):
    k.fn((g.blocks,), (THREADS_PER_BLOCK,), (
        g.ptr['luts'], st['tables'].data_ptr(), g.ptr['codes'], g.ptr['sign_mantissa'],
        g.ptr['output_positions'], g.ptr['gaps'], out_ptr, g.n_luts, g.n_bytes, g.n_elements),
        shared_mem=st['shared_mem'])


# abi name -> (prepare(kernel, gpu_case) -> state, launch(kernel, gpu_case, state, out_ptr))
#   v1:         the original kernel (git 4577338)
#   production: the current kernel, set up by the package's own host-side helpers
ABIS = {'v1': (_prepare_v1, _launch_v1), 'production': (_prepare_production, _launch_production)}


def register_abi(name, prepare, launch):
    ABIS[name] = (prepare, launch)


GUARD = 4096  # uint16 elements on each side of the output


def run_checked(kernel: Kernel, g: GpuCase, state=None):
    """Decode into a poisoned, guarded buffer. Returns (ok, message)."""
    state = kernel.prepare(g) if state is None else state
    n = g.n_elements
    buf = torch.full((n + 2 * GUARD,), 0x5A5A, dtype=torch.int16, device=g.expected.device)
    buf[GUARD:GUARD + n] = ~g.expected          # every element must be overwritten
    try:
        kernel.launch(g, state, buf.data_ptr() + 2 * GUARD)
    except cp.cuda.driver.CUDADriverError as e:
        smem = state if isinstance(state, int) else state.get('shared_mem', '?') if isinstance(state, dict) else '?'
        return False, f'launch failed: {str(e).split(":")[0]} (shared_mem={smem})'
    torch.cuda.synchronize()
    body = buf[GUARD:GUARD + n]
    if not bool((buf[:GUARD] == 0x5A5A).all()) or not bool((buf[GUARD + n:] == 0x5A5A).all()):
        return False, 'wrote outside the output buffer'
    bad = body != g.expected
    n_bad = int(bad.sum())
    if n_bad:
        first = int(torch.nonzero(bad)[0])
        return False, (f'{n_bad}/{n} elements wrong, first at {first}: '
                       f'got 0x{int(body[first]) & 0xFFFF:04x}, want 0x{int(g.expected[first]) & 0xFFFF:04x}')
    return True, 'ok'
