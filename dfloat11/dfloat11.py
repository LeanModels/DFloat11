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
import os
import re
import json
import pkg_resources
from sys import stderr
from typing import Optional, Dict, Union
from tqdm import tqdm

import torch
import torch.nn as nn

import cupy as cp
import numpy as np

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from .dfloat11_utils import get_codec, get_32bit_codec, get_luts, encode_weights


# Load CUDA kernel for custom DFloat11 tensor decoding
ptx_path = pkg_resources.resource_filename("dfloat11", "decode.ptx")
_decode = cp.RawModule(path=ptx_path).get_function('decode')


# Define constants
version = '0.5.0'
bytes_per_thread = 8
threads_per_block = (512, )
offloaded_tensor_names = (
    'encoded_exponent', 'sign_mantissa'
)


class TensorManager:
    """
    Static utility class that manages tensor allocation and reuse
    to minimize memory allocation overhead during tensor reconstruction.
    """
    # Static class variables to store tensors
    _tensors = {}  # Maps device to tensor

    @staticmethod
    def allocate_bfloat16(device, n_elements):
        """
        Get a bfloat16 tensor with at least n_elements on the specified device.

        If a tensor already exists on the device and is larger than n_elements,
        a slice of the tensor with exactly n_elements is returned. If n_elements 
        exceeds the size of the existing tensor, the existing tensor is deallocated 
        and a larger one is allocated.

        Args:
            device: The device to allocate the tensor on (e.g., 'cuda:0')
            n_elements: The exact number of elements required

        Returns:
            A bfloat16 tensor with exactly n_elements on the specified device
        """
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)
        
        # Check if we already have a tensor for this device
        if device in TensorManager._tensors:
            existing_tensor = TensorManager._tensors[device]
            
            # If existing tensor is large enough, return a slice of it
            if existing_tensor.numel() >= n_elements:
                return existing_tensor[:n_elements]
            
            # Otherwise, delete the existing tensor to free up memory
            del TensorManager._tensors[device]
            torch.cuda.empty_cache()  # Ensure memory is freed
        
        # Allocate a new tensor
        new_tensor = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
        print(f'Allocated {n_elements} bf16 on device {device}', file=stderr)
        
        # Store the tensor
        TensorManager._tensors[device] = new_tensor
        
        return new_tensor


def get_hook(threads_per_block, bytes_per_thread):
    """
    Creates a PyTorch forward pre-hook that decodes compressed DFloat11 weights on-the-fly.
    
    This hook reconstructs full-precision weights from compressed representations
    using a custom CUDA kernel during the forward pass.
    
    Args:
        threads_per_block: CUDA thread configuration 
        bytes_per_thread: Number of bytes processed per CUDA thread
        
    Returns:
        A forward pre-hook function for PyTorch modules
    """
    threads_per_block = tuple(threads_per_block)

    def decode_hook(module, _):
        device = module.luts.device

        # Load offloaded tensors to GPU if not already there
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name, tensor in module.offloaded_tensors.items():
                if not (
                    hasattr(module, tensor_name) and (getattr(module, tensor_name).device == device)
                ):
                    module.register_buffer(tensor_name, tensor.to(device, non_blocking=True))

        # Get dimensions for tensor reconstruction
        n_elements = module.sign_mantissa.numel()
        n_bytes = module.encoded_exponent.numel()
        n_luts = module.luts.shape[0]

        # Get output tensor for reconstructed weights
        reconstructed = TensorManager.allocate_bfloat16(device, n_elements)

        # Configure CUDA grid dimensions for the kernel launch
        blocks_per_grid = (int(math.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

        # Launch CUDA kernel to decode the compressed weights
        with cp.cuda.Device(device.index):
            _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=module.shared_mem_size, args=[
                module.luts.data_ptr(),
                module.encoded_exponent.data_ptr(),
                module.sign_mantissa.data_ptr(),
                module.output_positions.data_ptr(),
                module.gaps.data_ptr(),
                reconstructed.data_ptr(),
                n_luts, n_bytes, n_elements
            ])

        # Inject reconstructed weights into the appropriate module
        if isinstance(module, nn.Linear):
            module.weight = reconstructed.view(
                module.out_features, module.in_features
            )
        elif isinstance(module, nn.Embedding):
            module.weight = reconstructed.view(
                module.num_embeddings, module.embedding_dim
            )
        else:
            # Handle special case where weights need to be split across multiple submodules
            weights = torch.tensor_split(reconstructed, module.split_positions)
            for sub_module, weight in zip(module.weight_injection_modules, weights):
                sub_module.weight = weight.view(sub_module.out_features, sub_module.in_features)

        # Delete tensors from GPU if offloading is enabled
        if hasattr(module, 'offloaded_tensors'):
            for tensor_name in module.offloaded_tensors.keys():
                if hasattr(module, tensor_name):
                    tmp = getattr(module, tensor_name)
                    delattr(module, tensor_name)
                    del tmp

    return decode_hook


def load_and_replace_tensors(
    model,
    directory_path,
    dfloat11_config,
    cpu_offload=False,
    cpu_offload_blocks=None,
    pin_memory=True,
    from_single_file=False,
):
    """
    Loads DFloat11 compressed weights from safetensors files and configures the model
    to use them with on-the-fly decompression.
    
    Args:
        model: The PyTorch model to load weights into
        directory_path: Path to the directory containing safetensors files
        dfloat11_config: Configuration for DFloat11 compression
        
    Returns:
        The model with configured DFloat11 compression
    """
    threads_per_block = dfloat11_config['threads_per_block']
    bytes_per_thread  = dfloat11_config['bytes_per_thread']
    pattern_dict      = dfloat11_config['pattern_dict']
    
    # Get all .safetensors files in the directory
    safetensors_files = [
        f for f in os.listdir(directory_path) if f.endswith('.safetensors')
    ] if not from_single_file else [directory_path]
    loading_desc = 'Loading DFloat11 safetensors'
    if cpu_offload:
        loading_desc += ' (offloaded to CPU'
        if pin_memory:
            loading_desc += ', memory pinned'
        loading_desc += ')'

    for file_name in tqdm(safetensors_files, desc=loading_desc):
        file_path = os.path.join(directory_path, file_name) if not from_single_file else file_name
        
        # Load the tensors from the file
        loaded_tensors = load_file(file_path)
        
        # Iterate over each tensor in the file
        for tensor_name, tensor_value in loaded_tensors.items():
            # Check if this tensor exists in the model's state dict
            if tensor_name in model.state_dict():
                # Get the parameter or buffer
                if tensor_name in dict(model.named_parameters()):
                    # It's a parameter, we can set it directly
                    param = dict(model.named_parameters())[tensor_name]
                    if param.shape == tensor_value.shape:
                        param.data.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {param.shape} vs loaded {tensor_value.shape}", file=stderr)
                else:
                    # It's a buffer, we can also set it directly
                    buffer = dict(model.named_buffers())[tensor_name]
                    if buffer.shape == tensor_value.shape:
                        buffer.copy_(tensor_value)
                    else:
                        print(f"Shape mismatch for {tensor_name}: model {buffer.shape} vs loaded {tensor_value.shape}", file=stderr)
            else:
                # Split the tensor name to get module path
                parts = tensor_name.split('.')
                module = model
                
                # Navigate to the correct module
                for i, part in enumerate(parts[:-1]):
                    if hasattr(module, part):
                        module = getattr(module, part)
                    else:
                        print(f"Cannot find module path for {tensor_name}", file=stderr)
                        break
                else:
                    if parts[-1] == 'split_positions':
                        setattr(module, 'split_positions', tensor_value.tolist())
                    else:
                        if cpu_offload and (cpu_offload_blocks is None or cpu_offload_blocks > 0) and parts[-1] in offloaded_tensor_names:
                            if not hasattr(module, 'offloaded_tensors'):
                                setattr(module, 'offloaded_tensors', {})

                            module.offloaded_tensors[parts[-1]] = tensor_value.pin_memory() if pin_memory else tensor_value

                            if (cpu_offload_blocks is not None) and (cpu_offload_blocks > 0) and (len(module.offloaded_tensors) == len(offloaded_tensor_names)):
                                cpu_offload_blocks -= 1
                        else:
                            # Register the buffer to the found module
                            module.register_buffer(parts[-1], tensor_value)

                    # Set up decompression for encoded weights
                    if parts[-1] == 'encoded_exponent':
                        # Register the decode hook to decompress weights during forward pass
                        module.register_forward_pre_hook(get_hook(threads_per_block, bytes_per_thread))

                        # Configure weight injection based on module type
                        for pattern, attr_names in pattern_dict.items():
                            if re.fullmatch(pattern, '.'.join(parts[:-1])):
                                if isinstance(module, nn.Embedding):
                                    # Remove weight attribute from embedding layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                elif isinstance(module, nn.Linear):
                                    # Remove weight attribute from linear layer
                                    tmp = module.weight
                                    delattr(module, 'weight')
                                    del tmp
                                else:
                                    # Handle special case for multi-module weight injection
                                    setattr(module, 'weight_injection_modules', [])
                                    for attr_path in attr_names:
                                        parts = attr_path.split('.')
                                        target = module
                                        for p in parts:
                                            target = getattr(target, p)

                                        tmp = target.weight
                                        delattr(target, 'weight')
                                        del tmp
                                        module.weight_injection_modules.append(target)
                    elif parts[-1] == 'output_positions':
                        # Calculate required shared memory size for CUDA kernel
                        output_positions_np = tensor_value.view(torch.uint32).numpy()
                        setattr(
                            module,
                            'shared_mem_size',
                            threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
                        )
    
    return model


def get_no_split_classes(model, pattern_dict):
    """
    Find model layer classes that should not be split across devices.
    
    This is crucial for DFloat11 model sharding to ensure compressed modules
    stay on the same device as their decompression buffers.
    
    Args:
        model: The PyTorch model
        pattern_dict: Dictionary mapping regex patterns to submodule lists
        
    Returns:
        List of class names that should not be split across devices
    """
    no_split_classes = []
    for pattern in pattern_dict:
        for full_name, sub_module in model.named_modules():
            if re.fullmatch(pattern, full_name):
                class_name = sub_module.__class__.__name__
                if class_name not in no_split_classes:
                    no_split_classes.append(class_name)

    return no_split_classes


class DFloat11Model:
    """
    Wrapper class for loading and using models with DFloat11 compressed weights.
    DFloat11 is a custom 11-bit floating point format that provides memory efficiency
    while maintaining numerical accuracy for LLM weights.
    """
    @classmethod
    def from_pretrained(
        cls,
        dfloat11_model_name_or_path: str,
        device: Optional[str] = None,
        device_map: str = 'auto',
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        bfloat16_model = None,
        cpu_offload: bool = False,
        cpu_offload_blocks: Optional[int] = None,
        pin_memory: bool = True,
        from_single_file: bool = False,
        pattern_dict: Optional[dict[str, list[str]]] = None,
        **kwargs,
    ):
        """
        Load a model with DFloat11 compressed weights from local path or Hugging Face Hub.
        
        Args:
            dfloat11_model_name_or_path: Local path or HF Hub model name
            device: Target device for the model
            device_map: Strategy for distributing model across devices
            max_memory: Maximum memory allocation per device
            bfloat16_model: Optional pre-initialized model to load weights into
            cpu_offload: Enables CPU offloading; only keeps a single block of weights in GPU at once
            cpu_offload_blocks: Number of transformer blocks to offload to CPU; if None, offload all blocks
            pin_memory: Enables memory-pinning/page-locking when using CPU offloading
            from_single_file: Whether to load a single safetensors file
            pattern_dict: Dictionary mapping regex patterns to submodule lists
            **kwargs: Additional arguments passed to AutoModelForCausalLM.from_config
            
        Returns:
            Model with DFloat11 compressed weights configured for on-the-fly decompression
        """
        # Resolve model path, downloading from HF Hub if needed
        if from_single_file:
            if os.path.isfile(dfloat11_model_name_or_path):
                dfloat11_model_path = dfloat11_model_name_or_path
            elif os.path.isdir(dfloat11_model_name_or_path):
                raise IsADirectoryError(f'Expected `dfloat11_model_name_or_path` to be the path to a safetensors file, but found a directory: "{dfloat11_model_name_or_path}".')
            else:
                raise FileNotFoundError(f'The file "{dfloat11_model_name_or_path}" does not exist.')
        else:
            if os.path.exists(dfloat11_model_name_or_path):
                dfloat11_model_path = dfloat11_model_name_or_path
            else:
                dfloat11_model_path = dfloat11_model_name_or_path.replace('/', '__')
                if not os.path.exists(dfloat11_model_path):
                    snapshot_download(dfloat11_model_name_or_path, local_dir=dfloat11_model_path)

        # Load model configuration
        if bfloat16_model:
            if from_single_file:
                config = {
                    'dfloat11_config': {
                        'version': version,
                        'threads_per_block': threads_per_block,
                        'bytes_per_thread': bytes_per_thread,
                        'pattern_dict': pattern_dict,
                    },
                }
            else:
                with open(os.path.join(dfloat11_model_path, 'config.json'), 'r', encoding='utf-8') as f:
                    config = json.load(f)

            model = bfloat16_model
        else:
            from transformers import AutoModelForCausalLM, AutoConfig, GenerationConfig
            from transformers.modeling_utils import no_init_weights

            # Initialize model without loading weights
            config = AutoConfig.from_pretrained(dfloat11_model_path)
            with no_init_weights():
                model = AutoModelForCausalLM.from_config(
                    config, torch_dtype=torch.bfloat16, **kwargs,
                )
                model.tie_weights()
                model.eval()

            # Try to load generation config if available
            try:
                generation_config = GenerationConfig.from_pretrained(dfloat11_model_path)
                model.generation_config = generation_config
            except Exception as e:
                pass

        # Verify model has DFloat11 configuration
        if isinstance(config, dict) and 'dfloat11_config' in config:
            dfloat11_config = config['dfloat11_config']
        elif hasattr(config, 'dfloat11_config'):
            dfloat11_config = config.dfloat11_config
        else:
            raise AttributeError('"dfloat11_config" not found: it is expected to be found in the config file or passed as an argument.')

        # Load compressed weights and configure decompression
        load_and_replace_tensors(
            model, dfloat11_model_path, dfloat11_config,
            cpu_offload=cpu_offload, cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory, from_single_file=from_single_file,
        )

        if not cpu_offload:
            # Calculate and report model size
            model_bytes = 0
            for param in model.state_dict().values():
                model_bytes += param.nbytes

            print(f"Total model size: {model_bytes / 1e9:0.4f} GB", file=stderr)

        # Move model to specified device or distribute across multiple devices
        if device:
            model = model.to(device)
        else:
            assert device_map == 'auto', "device_map should be 'auto' if no specific device is provided."
            # Identify modules that must remain on same device for decompression
            no_split_classes = get_no_split_classes(model, dfloat11_config['pattern_dict'])
            max_memory = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            model = dispatch_model(model, device_map)

            # Warn if model is not fully on GPU
            if any(param.device.type == 'cpu' for param in model.parameters()):
                print("Warning: Some model layers are on CPU. For inference, ensure the model is fully loaded onto CUDA-compatible GPUs.", file=stderr)

        return model


    @classmethod
    def from_single_file(
        cls,
        dfloat11_model_path: str,
        pattern_dict: dict[str, list[str]],
        bfloat16_model,
        device: Optional[str] = None,
        device_map: str = 'auto',
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
        cpu_offload: bool = False,
        cpu_offload_blocks: Optional[int] = None,
        pin_memory: bool = True,
    ):
        return cls.from_pretrained(
            dfloat11_model_name_or_path=dfloat11_model_path,
            device=device,
            device_map=device_map,
            max_memory=max_memory,
            bfloat16_model=bfloat16_model,
            cpu_offload=cpu_offload,
            cpu_offload_blocks=cpu_offload_blocks,
            pin_memory=pin_memory,
            from_single_file=True,
            pattern_dict=pattern_dict,
        )


def compress_model(
    model,
    pattern_dict: dict[str, list[str]],
    save_path: str,
    block_range: list[int] = [0, 10000],
    save_single_file: bool = True,
    check_correctness: bool = True,
):
    os.makedirs(save_path, exist_ok=True)

    block_index = 0
    save_model = True
    # Iterate over each pattern and its associated attribute names
    for pattern, attr_names in pattern_dict.items():
        # Find all sub-modules whose full name matches the pattern
        for full_name, sub_module in model.named_modules():
            if re.fullmatch(pattern, full_name):
                if not save_single_file:
                    if '.' in full_name:
                        parent_name, child_name = full_name.rsplit('.', 1)
                        parent = model.get_submodule(parent_name)
                        setattr(parent, child_name, None)
                    else:
                        setattr(model, full_name, None)

                block_index += 1
                if block_index <= block_range[0]:
                    continue
                elif block_index > block_range[1]:
                    save_model = False
                    break

                weights = []
                if isinstance(sub_module, nn.Embedding):
                    assert sub_module.weight.data.dtype == torch.bfloat16, \
                        f"Expected weights to be in bfloat16 format for compression, but '{full_name}' has dtype {sub_module.weight.data.dtype}"
                    weights.append(sub_module.weight.data.detach().cpu().flatten())
                    delattr(sub_module, 'weight')
                elif isinstance(sub_module, nn.Linear):
                    assert sub_module.weight.data.dtype == torch.bfloat16, \
                        f"Expected weights to be in bfloat16 format for compression, but '{full_name}' has dtype {sub_module.weight.data.dtype}"
                    weights.append(sub_module.weight.data.detach().cpu().flatten())
                    delattr(sub_module, 'weight')
                else:
                    # Extract weights and replace sub-module attributes
                    for attr_path in attr_names:
                        parts = attr_path.split('.')
                        target = sub_module
                        for p in parts:
                            target = getattr(target, p)

                        assert target.weight.data.dtype == torch.bfloat16, \
                            f"Expected weights to be in bfloat16 format for compression, but '{full_name}.{attr_path}' has dtype {target.weight.data.dtype}"
                        weights.append(target.weight.data.detach().cpu().flatten())
                        delattr(target, 'weight')

                _codec, _counter = get_codec(torch.cat(weights))
                codec, _, table = get_32bit_codec(_counter)
                codec.print_code_table()

                luts = get_luts(table)

                encoded, other_8bits, output_positions, gaps, split_positions = encode_weights(weights, codec, bytes_per_thread, threads_per_block[0])

                if check_correctness:
                    device = torch.device('cuda:0')

                    n_luts = luts.shape[0]
                    n_elements = other_8bits.numel()
                    n_bytes = encoded.numel()

                    cuda_luts, cuda_encoded, cuda_other_8bits, cuda_output_positions, cuda_gaps = list(map(lambda x: x.to(device), [luts, encoded, other_8bits, output_positions, gaps]))
                    cuda_outputs = torch.empty(n_elements, dtype=torch.bfloat16, device=device)

                    blocks_per_grid = (int(np.ceil(n_bytes / (threads_per_block[0] * bytes_per_thread))), )

                    output_positions_np = output_positions.view(torch.uint32).numpy()
                    shared_mem_size = threads_per_block[0] * 4 + 4 + (output_positions_np[1:] - output_positions_np[:-1]).max().item() * 2
                    print(f'Using {shared_mem_size} bytes of shared memory.')

                    with cp.cuda.Device(device.index):
                        _decode(grid=blocks_per_grid, block=threads_per_block, shared_mem=shared_mem_size, args=[
                            cuda_luts.data_ptr(),
                            cuda_encoded.data_ptr(),
                            cuda_other_8bits.data_ptr(),
                            cuda_output_positions.data_ptr(),
                            cuda_gaps.data_ptr(),
                            cuda_outputs.data_ptr(),
                            n_luts, n_bytes, n_elements
                        ])

                    _is_correct = (torch.cat(weights) == cuda_outputs.cpu()).all().item()
                    if _is_correct:
                        print('✅ Correctness check passed: decompressed weights match the original weights.')
                    else:
                        raise RuntimeError(f'❌ Correctness check failed: The decompressed weights do not match the original weights for module "{full_name}".')

                sub_module.register_buffer('luts', luts)
                sub_module.register_buffer('encoded_exponent', encoded)
                sub_module.register_buffer('sign_mantissa', other_8bits)
                sub_module.register_buffer('output_positions', output_positions.view(torch.uint8))
                sub_module.register_buffer('gaps', gaps)
                sub_module.register_buffer('split_positions', split_positions)

                if not save_single_file:
                    state_dict = sub_module.state_dict()
                    state_dict = {f'{full_name}.{key}': value for key, value in state_dict.items()}

                    save_file(state_dict, os.path.join(save_path, full_name.replace('.', '_') + '.safetensors'))

    if save_model:
        dfloat11_config = {
            'version': version,
            'threads_per_block': threads_per_block,
            'bytes_per_thread': bytes_per_thread,
            'pattern_dict': pattern_dict,
        }
        if hasattr(model, 'config'):
            try:
                model.config.dfloat11_config = dfloat11_config
            except Exception:
                pass
        
        if hasattr(model, 'save_pretrained') and not save_single_file:
            model.save_pretrained(save_path)
        else:
            save_file(model.state_dict(), os.path.join(save_path, 'model.safetensors'))
        
        save_config = True
        if os.path.exists(os.path.join(save_path, 'config.json')):
            with open(os.path.join(save_path, 'config.json'), 'r', encoding='utf-8') as f:
                config = json.load(f)

                if 'dfloat11_config' in config and isinstance(config['dfloat11_config'], dict):
                    save_config = False

        if save_config:
            with open(os.path.join(save_path, 'config.json'), 'w') as config_file:
                json.dump({
                    'dfloat11_config': dfloat11_config,
                }, config_file, indent=2)
