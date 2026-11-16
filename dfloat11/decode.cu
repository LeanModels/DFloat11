// Copyright 2025 Tianyi Zhang
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// DFloat11 decoder: Huffman-coded bfloat16 exponents + raw sign/mantissa bytes -> bfloat16.
//
// Each block decodes BYTES_PER_THREAD * blockDim.x bytes of the exponent stream; each thread
// decodes the codes that *start* inside its own BYTES_PER_THREAD-byte window (skipping `gap`
// leading bits that belong to the previous window's last code). A thread first counts its codes,
// the block prefix-sums the counts to find where each thread's output starts, and the codes are
// decoded again into a shared-memory buffer of exponent bytes. Finally the block merges those with
// the sign/mantissa bytes and writes the bfloat16 values with wide, coalesced stores.
//
// Launch requirements (see dfloat11_utils.decode_shared_mem_size / get_decode_tables):
//   * blockDim.x is a multiple of 32 and at most 1024;
//   * dynamic shared memory of DECODE_SMEM_FIXED + (max elements of any block) + 8 bytes;
//   * `tables` holds the 512 bytes from get_decode_tables(luts), 4-byte aligned.
// Wide loads/stores are used when `codes` is 4-byte, `sign_mantissa` 8-byte and `outputs` 16-byte
// aligned; otherwise the kernel falls back to narrower accesses and gives the same result.

#define THREAD_ID           threadIdx.x
#define BLOCK_ID            blockIdx.x
#define N_THREADS           blockDim.x
#define BYTES_PER_THREAD    8
#define DECODE_SMEM_FIXED   640

typedef unsigned char       uint8_t;
typedef unsigned short      uint16_t;
typedef unsigned int        uint32_t;
typedef unsigned long long  uint64_t;

static __device__ __forceinline__ uint32_t bswap32(const uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}

// Loads codes[off, off + 12) as three big-endian words; bytes at or beyond n_bytes read as 0.
static __device__ __forceinline__ void load_window(
    const uint8_t * __restrict__ codes, const uint32_t off, const uint32_t n_bytes,
    uint32_t &w0, uint32_t &w1, uint32_t &w2
) {
    if ((((uint64_t) codes) & 3) == 0 && off + 12 <= n_bytes) {
        const uint32_t *p = (const uint32_t *) (codes + off);
        w0 = bswap32(__ldg(p));
        w1 = bswap32(__ldg(p + 1));
        w2 = bswap32(__ldg(p + 2));
    } else {
        uint32_t w[3] = {0, 0, 0};
        for (uint32_t i = 0; i < 12; i++) {
            if (off + i < n_bytes) {
                w[i >> 2] |= ((uint32_t) __ldg(&codes[off + i])) << (24 - 8 * (i & 3));
            }
        }
        w0 = w[0]; w1 = w[1]; w2 = w[2];
    }
}

// The 32 bits starting at bit `pos` (0 <= pos < 64) of the 96-bit string w0:w1:w2.
static __device__ __forceinline__ uint32_t peek32(
    const uint32_t w0, const uint32_t w1, const uint32_t w2, const uint32_t pos
) {
    const bool second = pos >= 32;
    return __funnelshift_l(second ? w2 : w1, second ? w1 : w0, pos);
}

// Full multi-level lookup of the code at the top of `bits`. Values >= 240 in `luts` point to the
// table for the next byte of the code.
static __device__ __forceinline__ uint32_t decode_symbol(const uint8_t * __restrict__ luts, const uint32_t bits) {
    uint32_t decoded = __ldg(&luts[bits >> 24]);
    if (decoded >= 240) {
        decoded = __ldg(&luts[256 * (256 - decoded) + ((bits >> 16) & 0xff)]);
        if (decoded >= 240) {
            decoded = __ldg(&luts[256 * (256 - decoded) + ((bits >> 8) & 0xff)]);
            if (decoded >= 240) {
                decoded = __ldg(&luts[256 * (256 - decoded) + (bits & 0xff)]);
            }
        }
    }
    return decoded;
}

static __device__ __forceinline__ uint32_t to_bfloat16_pair(const uint32_t sign_mantissa, const uint32_t exponent) {
    // Two bfloat16 values from two (sign|mantissa, exponent) byte pairs held in 16-bit lanes.
    return ((sign_mantissa & 0x00800080u) << 8) | (exponent << 7) | (sign_mantissa & 0x007f007fu);
}

extern "C"
__global__ void decode(
    const uint8_t * __restrict__    luts,
    const uint8_t * __restrict__    tables,
    const uint8_t * __restrict__    codes,
    const uint8_t * __restrict__    sign_mantissa,
    const uint32_t * __restrict__   position_offsets,
    const uint8_t * __restrict__    gaps,
    uint16_t * __restrict__         outputs,
    const int n_luts, const int n_bytes, const int n_elements
) {
    extern __shared__ uint8_t shared_mem[];
    uint32_t* warp_sums       = (uint32_t*) shared_mem;     // [32] per-warp symbol counts
    uint8_t*  short_lens      = shared_mem + 128;           // [256] see get_decode_tables
    uint8_t*  multi_counts    = shared_mem + 384;           // [256] see get_decode_tables
    uint8_t*  exponent_buffer = shared_mem + DECODE_SMEM_FIXED;
    const uint8_t * __restrict__ lens = luts + 256 * (n_luts - 1);
    const uint32_t global_thread_id = BLOCK_ID * N_THREADS + THREAD_ID;

    // Issue the global loads first so their latency overlaps the table copy.
    uint32_t w0, w1, w2;
    load_window(codes, global_thread_id * BYTES_PER_THREAD, n_bytes, w0, w1, w2);

    const uint32_t gap_bit = global_thread_id * 5;
    uint32_t gap_word = ((uint32_t) __ldg(&gaps[gap_bit >> 3])) << 8;
    if ((gap_bit & 7) > 3) {
        gap_word |= __ldg(&gaps[(gap_bit >> 3) + 1]);
    }
    const uint32_t gap = (gap_word >> (11 - (gap_bit & 7))) & 0x1f;

    // short_lens and multi_counts are adjacent in both places: copy them as 128 words.
    for (uint32_t i = THREAD_ID; i < 128; i += N_THREADS) {
        ((uint32_t *) short_lens)[i] = __ldg(&((const uint32_t *) tables)[i]);
    }
    __syncthreads();

    // Pass 1: count the codes that start in this thread's 64 bits. While pos <= 56, every code
    // that lies entirely inside the next 8 bits also starts before bit 64, so one lookup in
    // multi_counts can count several short codes at once.
    uint32_t thread_counter = 0;
    uint32_t pos = gap;
    while (pos <= 56) {
        const uint32_t bits = peek32(w0, w1, w2, pos);
        const uint32_t entry = multi_counts[bits >> 24];
        if (entry != 0) {
            thread_counter += entry >> 4;
            pos += entry & 15;
        } else {
            thread_counter += 1;
            pos += max((uint32_t) __ldg(&lens[decode_symbol(luts, bits)]), 1u);
        }
    }
    while (pos < 64) {
        const uint32_t bits = peek32(w0, w1, w2, pos);
        uint32_t len = short_lens[bits >> 24];
        if (len == 0) {
            len = max((uint32_t) __ldg(&lens[decode_symbol(luts, bits)]), 1u);
        }
        thread_counter += 1;
        pos += len;
    }

    // Exclusive prefix sum of the counts: a shuffle scan within each warp, then warp 0 scans the
    // per-warp totals.
    const uint32_t lane = THREAD_ID & 31, warp = THREAD_ID >> 5;
    uint32_t inclusive = thread_counter;
    for (uint32_t d = 1; d < 32; d <<= 1) {
        const uint32_t up = __shfl_up_sync(0xffffffffu, inclusive, d);
        if (lane >= d) inclusive += up;
    }
    if (lane == 31) {
        warp_sums[warp] = inclusive;
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t total = lane < (N_THREADS >> 5) ? warp_sums[lane] : 0;
        uint32_t scan = total;
        for (uint32_t d = 1; d < 32; d <<= 1) {
            const uint32_t up = __shfl_up_sync(0xffffffffu, scan, d);
            if (lane >= d) scan += up;
        }
        if (lane < (N_THREADS >> 5)) {
            warp_sums[lane] = scan - total;
        }
    }
    __syncthreads();

    const uint32_t write_offset = position_offsets[BLOCK_ID];
    const uint32_t block_end = min(position_offsets[BLOCK_ID + 1], (uint32_t) n_elements);
    // exponent_buffer[k] holds element base + k; aligning base to 8 lets the copy-out below use
    // 8-byte shared loads and 16-byte global stores.
    const uint32_t base = write_offset & ~7u;
    uint32_t output_idx = write_offset + warp_sums[warp] + inclusive - thread_counter;
    const uint32_t end_output_idx = min(output_idx + thread_counter, (uint32_t) n_elements);

    // Pass 2: decode. For codes of up to 8 bits the symbol and its length come from two
    // independent lookups; longer codes take the multi-level path.
    pos = gap;
    while (output_idx < end_output_idx) {
        const uint32_t bits = peek32(w0, w1, w2, pos);
        uint32_t decoded = __ldg(&luts[bits >> 24]);
        uint32_t len = short_lens[bits >> 24];
        if (len == 0) {
            decoded = decode_symbol(luts, bits);
            len = max((uint32_t) __ldg(&lens[decoded]), 1u);
        }
        exponent_buffer[output_idx - base] = decoded;
        output_idx += 1;
        pos += len;
    }
    __syncthreads();

    // Merge exponents with sign/mantissa bytes, 8 elements per thread per iteration.
    const bool aligned = ((((uint64_t) outputs) & 15) | (((uint64_t) sign_mantissa) & 7)) == 0;
    const uint2 * exponent_words = (const uint2 *) exponent_buffer;
    for (uint32_t w = THREAD_ID; base + 8 * w < block_end; w += N_THREADS) {
        const uint32_t i = base + 8 * w;
        if (aligned && i >= write_offset && i + 8 <= block_end) {
            const uint2 e  = exponent_words[w];
            const uint2 sm = __ldg((const uint2 *) (sign_mantissa + i));
            uint4 out;
            out.x = to_bfloat16_pair(__byte_perm(sm.x, 0, 0x4140), __byte_perm(e.x, 0, 0x4140));
            out.y = to_bfloat16_pair(__byte_perm(sm.x, 0, 0x4342), __byte_perm(e.x, 0, 0x4342));
            out.z = to_bfloat16_pair(__byte_perm(sm.y, 0, 0x4140), __byte_perm(e.y, 0, 0x4140));
            out.w = to_bfloat16_pair(__byte_perm(sm.y, 0, 0x4342), __byte_perm(e.y, 0, 0x4342));
            *(uint4 *) (outputs + i) = out;
        } else {
            for (uint32_t j = max(i, write_offset); j < min(i + 8, block_end); j++) {
                const uint32_t sm = __ldg(&sign_mantissa[j]);
                outputs[j] = ((sm & 0x80) << 8) | (((uint32_t) exponent_buffer[j - base]) << 7) | (sm & 0x7f);
            }
        }
    }
}
