# Copyright 2025 Tianyi Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np

from dahuffman import HuffmanCodec

from tqdm import tqdm
from copy import copy


def get_codec(weight):
    counter = {}
    W = weight.view(torch.int16)
    exponent_8bits = (W >> 7) & 0xFF
    vals, freqs = torch.unique(exponent_8bits, return_counts=True)
    for v, f in zip(vals.tolist(), freqs.tolist()):
        counter[v] = f
    
    codec = HuffmanCodec.from_frequencies(counter)

    return codec, counter


def get_32bit_codec(counter):
    codec = HuffmanCodec.from_frequencies(counter)
    table = codec.get_code_table()
    max_len = 0
    for _, (l, _) in table.items():
        max_len = max(max_len, l)

    compressed_codec = codec
    compressed_counter = counter

    min_k = 2
    freq = np.array(list(counter.values()))
    while max_len > 32:
        min_indices = np.argpartition(freq, min_k)[:min_k]
        min_k += 1
        min_keys = np.array(list(counter.keys()))[min_indices]

        compressed_counter = copy(counter)
        for k in min_keys:
            compressed_counter[k] = 1
        compressed_codec = HuffmanCodec.from_frequencies(compressed_counter)
        table = compressed_codec.get_code_table()
        max_len = 0
        for _, (l, _) in table.items():
            max_len = max(max_len, l)

        print(min_k - 1, max_len)

    return compressed_codec, compressed_counter, table


def get_luts(table):
    prefixes = ['']

    for key, (bits, val) in table.items():
        if isinstance(key, int):
            prefix = bin(val)[2:].rjust(bits, "0")[:((bits - 1) // 8 * 8)]
            if prefix not in prefixes:
                prefixes.append(prefix)

    prefixes.sort(key=len)

    luts = np.zeros((len(prefixes), 256), dtype=np.uint8)

    for pi, p in enumerate(prefixes):
        bytes_dict = {}
        pl = len(p) // 8
        for key, (bits, val) in table.items():
            if isinstance(key, int):
                bin_val = bin(val)[2:].rjust(bits, '0')

                if bin_val.startswith(p):
                    if (bits - 1) // 8 == pl:
                        dict_key = int(bin_val[(pl * 8):].ljust(8, '0'), 2)
                        dict_value = key
                    else:
                        dict_key = int(bin_val[(pl * 8):(pl * 8 + 8)], 2)
                        dict_value = 256 - prefixes.index(bin_val[:(pl * 8 + 8)])

                    if dict_key in bytes_dict and bytes_dict[dict_key] != dict_value:
                        raise ValueError(f"Key {dict_key} already exists in {bytes_dict}")
                    else:
                        bytes_dict[dict_key] = dict_value

        print(bytes_dict)

        for i in range(256):
            if i in bytes_dict:
                curr_val = bytes_dict[i]
            luts[pi, i] = curr_val

    lens = np.zeros((1, 256), dtype=np.uint8)
    for key, (bits, val) in table.items():
        if isinstance(key, int):
            lens[-1, key] = bits

    return torch.from_numpy(
        np.concatenate((luts, lens), axis=0)
    )


def encode(data, codec, bytes_per_thread, threads_per_block):
    encoded = []

    gaps = []
    output_positions = []

    buffer = 0
    size = 0
    total_size = 0
    element_count = 0
    for s in tqdm(data):
        if total_size // (8 * bytes_per_thread) + 1 > len(gaps):
            gaps.append(total_size - total_size // (8 * bytes_per_thread) * (8 * bytes_per_thread))

        if total_size // (8 * bytes_per_thread * threads_per_block) + 1 > len(output_positions):
            output_positions.append(element_count)

        b, v = codec._table[s]
        buffer = (buffer << b) + v
        size += b
        total_size += b
        element_count += 1
        while size >= 8:
            byte = buffer >> (size - 8)
            encoded.append(byte)
            buffer = buffer - (byte << (size - 8))
            size -= 8

    if size > 0:
        if total_size // (8 * bytes_per_thread) + 1 > len(gaps):
            gaps.append(total_size - total_size // (8 * bytes_per_thread) * (8 * bytes_per_thread))

        if total_size // (8 * bytes_per_thread * threads_per_block) + 1 > len(output_positions):
            output_positions.append(element_count)

        b, v = codec._table[codec._eof]
        buffer = (buffer << b) + v
        size += b
        if size >= 8:
            byte = buffer >> (size - 8)
        else:
            byte = buffer << (8 - size)
        encoded.append(byte)

    output_positions.append(len(data))

    blocks_per_grid = int(np.ceil(len(encoded) / (threads_per_block * bytes_per_thread)))

    gaps.extend([0] * (threads_per_block * blocks_per_grid - len(gaps)))
    binary_str_gaps = [format(gap, '05b') for gap in gaps]
    binary_gaps = [int(bit) for binary in binary_str_gaps for bit in binary]

    return np.frombuffer(bytes(encoded), dtype=np.uint8), np.packbits(binary_gaps), np.array(output_positions, dtype=np.uint32)


def encode_weights(weights, codec, bytes_per_thread, threads_per_block):
    split_positions = torch.cumsum(torch.LongTensor(list(map(lambda x: x.numel(), weights))), dim=0)[:-1]
    W_combined = torch.cat(weights).view(torch.int16)

    exponent_8bits = ((W_combined >> 7) & 0xFF).to(torch.uint8)
    other_8bits    = ((W_combined >> 8) & 0x80 | (W_combined & 0x7F)).to(torch.uint8)

    encoded, gaps, output_positions = list(map(
        torch.from_numpy,
        encode(exponent_8bits.tolist(), codec, bytes_per_thread, threads_per_block)
    ))
    
    print(f'Compression factor: {(encoded.numel() + other_8bits.numel() + output_positions.numel() * 4 + gaps.numel()) * 100 / (W_combined.numel() * 2):.2f}%')

    return encoded, other_8bits, output_positions, gaps, split_positions
