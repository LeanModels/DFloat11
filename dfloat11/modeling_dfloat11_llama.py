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

import math
import pickle
import os
from tqdm import tqdm
from sys import stderr

import torch
import torch.nn as nn
import torch.nn.functional as F

import cupy as cp

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from typing import Callable, Optional, Tuple, Dict, Union
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

# CUDA kernel execution configuration
threads_per_block = (256, )
bytes_per_thread = 8


class TensorManager:
    # Static class variable to store tensors for each device
    _tensors = {}  # Maps device to tensor

    @staticmethod
    def get_tensor(device, n_elements):
        """
        Get a bfloat16 tensor with at least n_elements on the specified device.

        If a tensor already exists on the device and is larger than n_elements,
        a slice of the tensor with exactly n_elements is returned. If n_elements 
        exceeds the size of the existing tensor, the existing tensor is deallocated 
        and a larger one is allocated.

        This tensor will be used to hold losslessly decompressed BFloat16 weights.
        """
        if isinstance(device, str):
            device = torch.device(device)

        if device in TensorManager._tensors:
            existing_tensor = TensorManager._tensors[device]
            if existing_tensor.numel() >= n_elements:
                return existing_tensor[:n_elements]
            del TensorManager._tensors[device]
            torch.cuda.empty_cache()

        new_tensor = torch.zeros(n_elements, dtype=torch.bfloat16, device=device)
        TensorManager._tensors[device] = new_tensor
        return new_tensor

    @staticmethod
    def clear_device(device=None):
        """
        Clear tensors for a specific device or all devices if none specified.
        """
        if device is None:
            TensorManager._tensors.clear()
        else:
            if isinstance(device, str):
                device = torch.device(device)
            if device in TensorManager._tensors:
                del TensorManager._tensors[device]

        torch.cuda.empty_cache()


class PlaceHolderLinear(nn.Module):
    """
    Placeholder linear layer to allow on-the-fly injection of losslessly decoded weights.
    """
    def __init__(self, original_linear):
        super(PlaceHolderLinear, self).__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.weight = None
        if isinstance(original_linear.bias, torch.Tensor):
            self.register_buffer('bias', original_linear.bias.data)
        else:
            self.bias = None

    def store_weight(self, weight):
        """Store a reference to the weight."""
        self.weight = weight.view(self.out_features, self.in_features)

    def forward(self, x):
        if self.weight is None:
            raise RuntimeError("Weight must be stored using `store_weight` before calling forward.")

        out = F.linear(x, self.weight, self.bias)
        self.weight = None  # Immediately discard to save memory
        return out


def replace_linear_layers(module):
    """
    Recursively replaces all nn.Linear layers with PlaceHolderLinear layers.
    This enables lazy, lossless decompression of weights from DFloat11.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            placeholder = PlaceHolderLinear(child)
            del child
            setattr(module, name, placeholder)
        else:
            replace_linear_layers(child)

    return module


class DFloat11LlamaDecoderLayer(nn.Module):
    """
    Decoder layer variant that reconstructs lossless BFloat16 weights
    from compressed DFloat11 format at each forward pass.
    Weights are discarded immediately after use to save memory.
    """
    def __init__(
        self,
        decoder_layer: LlamaDecoderLayer,
        compressed_exponent: torch.Tensor,
        sign_mantissa: torch.Tensor,
        output_positions: torch.Tensor,
        gaps: torch.Tensor,
        split_positions: torch.Tensor,
        dfloat11_decode: Callable,
    ):
        super().__init__()
        self.hidden_size = decoder_layer.hidden_size

        self.register_buffer('compressed_exponent', compressed_exponent)
        self.register_buffer('sign_mantissa', sign_mantissa)
        self.register_buffer('output_positions', output_positions)
        self.register_buffer('gaps', gaps)
        self.split_positions = split_positions.tolist()

        self.self_attn = replace_linear_layers(decoder_layer.self_attn)
        self.mlp = replace_linear_layers(decoder_layer.mlp)
        self.input_layernorm = decoder_layer.input_layernorm
        self.post_attention_layernorm = decoder_layer.post_attention_layernorm

        self.dfloat11_decode = dfloat11_decode

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        n_elements = self.sign_mantissa.numel()
        n_bytes = self.compressed_exponent.numel()

        device = hidden_states.device
        reconstructed = TensorManager.get_tensor(device, n_elements)

        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

        with cp.cuda.Device(device.index):
            self.dfloat11_decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=1024 + 256 + threads_per_block[0] * (bytes_per_thread + 4) + 8 + threads_per_block[0] * bytes_per_thread * 4 * 2, args=[
                self.compressed_exponent.data_ptr(),
                self.sign_mantissa.data_ptr(),
                self.output_positions.data_ptr(),
                self.gaps.data_ptr(),
                reconstructed.data_ptr(),
                bytes_per_thread, n_bytes, n_elements
            ])

        q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, up_proj_weight, gate_proj_weight, down_proj_weight = torch.tensor_split(
            reconstructed, self.split_positions
        )

        self.self_attn.q_proj.store_weight(q_proj_weight)
        self.self_attn.k_proj.store_weight(k_proj_weight)
        self.self_attn.v_proj.store_weight(v_proj_weight)
        self.self_attn.o_proj.store_weight(o_proj_weight)

        self.mlp.gate_proj.store_weight(gate_proj_weight)
        self.mlp.up_proj.store_weight(up_proj_weight)
        self.mlp.down_proj.store_weight(down_proj_weight)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class DFloat11Linear(nn.Module):
    """
    Linear layer with weights stored in DFloat11 format.
    Weights are losslessly decompressed to BFloat16 each forward pass.
    """
    def __init__(
        self,
        linear: nn.Linear,
        compressed_exponent: torch.Tensor,
        sign_mantissa: torch.Tensor,
        output_positions: torch.Tensor,
        gaps: torch.Tensor,
        dfloat11_decode: Callable,
    ):
        super(DFloat11Linear, self).__init__()

        assert sign_mantissa.numel() == linear.in_features * linear.out_features

        self.in_features = linear.in_features
        self.out_features = linear.out_features

        self.weight = None
        if linear.bias:
            self.register_buffer('bias', linear.bias.data)
        else:
            self.bias = None

        self.register_buffer('compressed_exponent', compressed_exponent)
        self.register_buffer('sign_mantissa', sign_mantissa)
        self.register_buffer('output_positions', output_positions)
        self.register_buffer('gaps', gaps)

        self.dfloat11_decode = dfloat11_decode

    def forward(self, x):
        n_elements = self.sign_mantissa.numel()
        n_bytes = self.compressed_exponent.numel()

        device = x.device
        weight = TensorManager.get_tensor(device, n_elements)

        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

        with cp.cuda.Device(device.index):
            self.dfloat11_decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=1024 + 256 + threads_per_block[0] * (bytes_per_thread + 4) + 8 + threads_per_block[0] * bytes_per_thread * 4 * 2, args=[
                self.compressed_exponent.data_ptr(),
                self.sign_mantissa.data_ptr(),
                self.output_positions.data_ptr(),
                self.gaps.data_ptr(),
                weight.data_ptr(),
                bytes_per_thread, n_bytes, n_elements
            ])

        return F.linear(x, weight.view(self.out_features, self.in_features), self.bias)


class DFloat11ModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        dfloat11_path: str,
        attn_implementation: Optional[str] = None,
        device: Optional[str] = None,
        device_map: str = 'auto',
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    ):
        # Load standard HuggingFace CausalLM model in bf16
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )

        ptx_path = os.path.join(dfloat11_path, 'decode.ptx')
        if not os.path.exists(ptx_path):
            raise FileNotFoundError(f"Missing PTX kernel file: {ptx_path}. Please ensure the directory contains decode.ptx.")

        _dfloat11_decode = cp.RawModule(path=ptx_path).get_function('huffman_decode')

        # Load and replace lm_head
        lm_head_path = os.path.join(dfloat11_path, 'lm_head.pkl')
        if not os.path.exists(lm_head_path):
            raise FileNotFoundError(f"Missing file: {lm_head_path}. Please ensure it exists.")

        with open(lm_head_path, 'rb') as file:
            comp_dict = pickle.load(file)

        lm_head = model.lm_head
        model.lm_head = DFloat11Linear(
            lm_head,
            comp_dict['compressed_exponent'],
            comp_dict['sign_mantissa'],
            comp_dict['output_positions'],
            comp_dict['gaps'],
            _dfloat11_decode,
        )
        del lm_head  # Free memory

        # Load and replace each decoder layer
        for i in tqdm(range(len(model.model.layers)), desc='Loading DFloat11 weights'):
            layer_path = os.path.join(dfloat11_path, f'layer_{i}.pkl')
            if not os.path.exists(layer_path):
                raise FileNotFoundError(f"Missing file: {layer_path}. Please ensure all layer_*.pkl files are present.")

            with open(layer_path, 'rb') as file:
                comp_dict = pickle.load(file)

            decoder_layer = model.model.layers[i]
            model.model.layers[i] = DFloat11LlamaDecoderLayer(
                decoder_layer,
                comp_dict['compressed_exponent'],
                comp_dict['sign_mantissa'],
                comp_dict['output_positions'],
                comp_dict['gaps'],
                comp_dict['split_positions'],
                _dfloat11_decode,
            )
            del decoder_layer

        # Estimate and print model size (non-DFloat11 parameters only)
        model_bytes = 0
        for param in model.state_dict().values():
            if param.dtype in [torch.uint8, torch.int8]:
                model_bytes += param.numel()
            elif param.dtype in [torch.float16, torch.bfloat16, torch.int16, torch.uint16]:
                model_bytes += param.numel() * 2
            elif param.dtype in [torch.float32, torch.int32, torch.uint32]:
                model_bytes += param.numel() * 4
            elif param.dtype in [torch.float64, torch.int64, torch.uint64]:
                model_bytes += param.numel() * 8
            else:
                print(f'Unrecognized parameter data type {param.dtype}.', file=stderr)

        print(f"Total model size: {model_bytes / 1e9:0.4f} GB", file=stderr)

        # Move model to device or use accelerate dispatch
        if isinstance(device, str):
            model = model.to(device)
        else:
            assert device_map == 'auto', "device_map should be 'auto' if no specific device is provided."
            no_split_classes = [type(model.model.layers[0]).__name__]
            max_memory = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            model = dispatch_model(model, device_map)

        # Warn if model is not fully on GPU
        if any(param.device.type == 'cpu' for param in model.parameters()):
            print("Warning: Some model layers are on CPU. For inference, ensure the model is fully loaded onto CUDA-compatible GPUs.")

        return model
