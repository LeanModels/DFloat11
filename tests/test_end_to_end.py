"""End-to-end check of the package's host code around the decode kernel.

    python tests/test_end_to_end.py

Compresses a small model with compress_model() (which runs its own GPU
correctness check), loads it back with DFloat11Model.from_single_file() with
and without CPU offloading, and requires every decompressed weight and the
model output to be bit-identical to the original.
"""

import copy
import os
import sys
import tempfile

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dfloat11.dfloat11 import DFloat11Model, compress_model  # noqa: E402


class MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=True)

    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x)))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.mlp = MLP(d, h)

    def forward(self, x):
        return x + self.mlp(self.q(x))


class Tiny(nn.Module):
    def __init__(self, vocab=300, d=192, h=520, n_blocks=3):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_blocks)])
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


PATTERNS = {
    r'embed': [],
    r'blocks\.\d+': ['q', 'mlp.up', 'mlp.down'],
    r'head': [],
}


def make_model(seed):
    torch.manual_seed(seed)
    m = Tiny()
    with torch.no_grad():
        for i, p in enumerate(m.parameters()):
            p.normal_(0, 0.02 * (1 + i % 3))     # a few different exponent distributions
    return m.to(torch.bfloat16).eval()


def main():
    reference = make_model(0)
    original = {k: v.clone() for k, v in reference.state_dict().items()}
    ids = torch.randint(0, 300, (4, 33))
    with torch.no_grad():
        want = reference.to('cuda')(ids.cuda())

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        compress_model(copy.deepcopy(reference).cpu(), PATTERNS, tmp, save_single_file=True, check_correctness=True)
        path = os.path.join(tmp, 'model.safetensors')

        for kw in (dict(), dict(cpu_offload=True), dict(cpu_offload=True, cpu_offload_blocks=2, pin_memory=False)):
            skeleton = make_model(1)        # different weights: everything must come from the file
            model = DFloat11Model.from_single_file(path, PATTERNS, skeleton, device='cuda', **kw)
            # Decoded weights share one buffer per device (TensorManager), so snapshot each leaf's
            # weight when that leaf runs, i.e. while its parent's decode is still the live one.
            seen = {}
            leaves = {'embed': model.embed, 'head': model.head,
                      **{f'blocks.{i}.{n}': b.get_submodule(n)
                         for i, b in enumerate(model.blocks) for n in ('q', 'mlp.up', 'mlp.down')}}
            handles = [m.register_forward_pre_hook(lambda mod, _, name=name: seen.__setitem__(name, mod.weight.cpu().clone()))
                       for name, m in leaves.items()]
            with torch.no_grad():
                got = model(ids.cuda())
            for h in handles:
                h.remove()
            same_output = torch.equal(got, want)
            bad = [n for n in leaves if n not in seen or not torch.equal(seen[n], original[n + '.weight'])]
            same_bias = torch.equal(model.blocks[0].mlp.down.bias.cpu(), original['blocks.0.mlp.down.bias'])
            ok = same_output and not bad and same_bias
            if not same_output:
                print('  model output differs from the original model')
            print(f'{"PASS" if ok else "FAIL"} load {kw or "(no offload)"}' + (f': mismatched {bad}' if bad else ''))
            failures += not ok
            del model
            torch.cuda.empty_cache()

    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
