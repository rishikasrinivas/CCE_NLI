# coding=utf-8
# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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
import itertools
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

from .cache_utils import Cache
from .configuration_utils import PretrainedConfig
from .utils.generic import GeneralInterface
from .utils.import_utils import is_torch_flex_attn_available, is_torch_greater_or_equal, is_torchdynamo_compiling


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size  # noqa: N811
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    # Register a fake type to avoid crashing for annotations and `isinstance` checks
    BlockMask = torch.Tensor

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)

if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex


def and_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_ones((), dtype=torch.bool)
        for mask in mask_functions:
            result = result & mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return and_mask


def or_masks(*mask_functions: list[Callable]) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        result = q_idx.new_zeros((), dtype=torch.bool)
        for mask in mask_functions:
            result = result | mask(batch_idx, head_idx, q_idx, kv_idx)
        return result

    return or_mask



def create_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an HybridCache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # If we have an HybridCache structure, here we want to create the mask for the full layers
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, layer_idx
    )
    if early_exit:
        return attention_mask

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not past_key_values.is_compileable if past_key_values is not None else True

    # Allow slight deviations from causal mask
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def create_sliding_window_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an HybridCache structure, this
    function will return the mask corresponding to one of the "sliding_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
    """
    # If we have an HybridCache structure, here we want to create the mask for the sliding layers
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, layer_idx
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not past_key_values.is_compileable if past_key_values is not None else True

    # Allow slight deviations from sliding causal mask
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=sliding_window,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


def create_chunked_causal_mask(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
) -> Optional[Union[torch.Tensor, BlockMask]]:
    """
    Create a chunked attention causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Llama4. If `past_key_values` has an HybridCache structure, this
    function will return the mask corresponding to one of the "chunked_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
    """
    # If we have an HybridCache structure, here we want to create the mask for the sliding layers
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, kv_length, kv_offset = _preprocess_mask_arguments(
        config, input_embeds, attention_mask, cache_position, past_key_values, layer_idx
    )
    if early_exit:
        return attention_mask

    chunk_size = getattr(config, "attention_chunk_size", None)
    if chunk_size is None:
        raise ValueError("Could not find an `attention_chunk_size` argument in the config, or it is not set")

    # Raise if using chunked attention on context too large with FA2
    if config._attn_implementation == "flash_attention_2" and kv_length + kv_offset > chunk_size:
        raise ValueError(
            "Flash attention 2 cannot handle chunked attention, and the key-value length is larger than the chunk size so the "
            "chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    batch_size, dtype = input_embeds.shape[0], input_embeds.dtype
    mask_factory_function = chunked_causal_mask_function(chunk_size)
    mask_interface = ModifiedLlamaAttention

    # Do not allow skip if we are compiling (this is to match BC)
    # TODO: cyril -> probably revisit and remove this, but a lot of tests rely on it
    allow_is_causal_skip = not past_key_values.is_compileable if past_key_values is not None else True

    # Allow slight deviations from chunked causal mask
    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False

    # We now create the mask
    causal_mask = mask_interface(
        batch_size=batch_size,
        cache_position=cache_position,
        kv_length=kv_length,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=chunk_size,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
    )
    return causal_mask


LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
}


def create_masks_for_generate(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
    **kwargs,
):
    """
    This function mimics how we create the masks in the `modeling_xxx.py` files, and is used in `generate` in order
    to easily create the masks in advance, when we compile the forwards with Static caches.

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # The attribute reside in the text config for composite models
    effective_config = config.get_text_config()
    # Prepare the mask args
    mask_kwargs = {
        "config": effective_config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
    }

    # If the attribute exist, we need several masks
    if hasattr(effective_config, "layer_types"):
        causal_masks = {}
        for layer_pattern in set(effective_config.layer_types):
            causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
        return causal_masks
    # In this case, all layers are sliding
    elif getattr(effective_config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    # In this case, all layers are chunked
    elif getattr(effective_config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    # All layers use standard causal attention
    return create_causal_mask(**mask_kwargs)


# Below are utilities to pretty-print the different masks
# Print the matrix with words as row labels
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BLACK_SQUARE = "■"
WHITE_SQUARE = "⬚"
GREY_SQUARE = "∙"
LOW_TRIANGLE = "⬕"
UPPER_TRIANGLE = "⬔"


def get_style(style):
    if style == "majong":
        BLACK_SQUARE = "🀞"  # Full block (represents "on" or active)
        BLACK_SQUARE = "🀙"  # Full block (represents "on" or active)
        WHITE_SQUARE = "🀆"  # "▒"  # Light shade (represents "off" or inactive)
        LOW_TRIANGLE = "🀛"  # Lower left triangle (stylized indication)
        UPPER_TRIANGLE = "🀛"  # Upper left triangle (stylized indication)
    else:
        BLACK_SQUARE = "█"  # Full block (represents "on" or active)
        WHITE_SQUARE = "░"  # "▒"  # Light shade (represents "off" or inactive)
        LOW_TRIANGLE = "▙"  # Lower left triangle (stylized indication))
        UPPER_TRIANGLE = "▜"  # Upper left triangle (stylized indication)

    return BLACK_SQUARE, WHITE_SQUARE, LOW_TRIANGLE, UPPER_TRIANGLE


# LOW_TRIANGLE = UPPER_TRIANGLE = "⟍"   # Upper right triangle (stylized indication)

YELLOW_SQUARE = f"{YELLOW}{BLACK_SQUARE}{RESET}"
GREEN_SQUARE = f"{GREEN}{BLACK_SQUARE}{RESET}"


def tensor_to_mask_visual(original_tensor: torch.Tensor, grid_size=(20, 40), style="majong") -> str:
    BLACK_SQUARE, WHITE_SQUARE, LOW_TRIANGLE, UPPER_TRIANGLE = get_style(style)
    h, w = original_tensor.shape
    max_h, max_w = grid_size
    if not (h < max_h and w < max_w):
        # Preserve aspect ratio within max grid size
        aspect_ratio = 2 * w / h
        if aspect_ratio > 1:
            w = max_w
            h = min(max_h, max(1, round(max_w / aspect_ratio)))
        else:
            h = max_h
            w = max(1, round(max_h * aspect_ratio))

        # Step 1: Rescale tensor by average pooling
        tensor = original_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        tensor = F.adaptive_avg_pool2d(tensor, output_size=(h, w))[0, 0]  # Remove extra dims
    else:
        tensor = original_tensor

    # Step 3: Build the string representation
    result = []
    for i in range(h):
        row = ""
        for j in range(w):
            if tensor[i, j] == 1:
                row += BLACK_SQUARE
            elif tensor[i, j] == 0:
                row += WHITE_SQUARE
            else:
                if j > 0:
                    if tensor[i, j - 1] == 1:
                        row += LOW_TRIANGLE
                    elif tensor[i, j - 1] == 0:
                        row += UPPER_TRIANGLE
                    else:
                        row += BLACK_SQUARE if tensor[i, j] == 1 else WHITE_SQUARE
                else:
                    row += (
                        BLACK_SQUARE
                        if tensor[i, j] == 1
                        else (
                            WHITE_SQUARE
                            if tensor[i, j] == 0
                            else (UPPER_TRIANGLE if tensor[i, j + 1] == 1 else LOW_TRIANGLE)
                        )
                    )
        result.append(row)

    return "\n".join(result)


class AttentionMask(torch.Tensor):
    def __new__(cls, data, style=None):
        # Create a new instance of AttentionMask as a Tensor
        cls.style = style
        return torch.Tensor._make_subclass(cls, data, require_grad=False)

    def __init__(self, data):
        # You can initialize any additional metadata here if needed
        pass

    def to_string(self, grid_size=(20, 40), limit=4):
        """Returns a string representation of the block mask."""
        dense_mask = self
        *batch_dims, num_rows, num_cols = dense_mask.shape
        total_vis = []

        for idx, batch_idx in enumerate(itertools.product(*[range(i) for i in batch_dims])):
            if idx == limit:
                total_vis.append("...")
                total_vis.append("To print out more, set AttentionMask.to_string(limit=N)")
                total_vis.append("You can also index (AttentionMask[batch, head]) to choose a specific batch or head")
                break
            block_vis = tensor_to_mask_visual(dense_mask[batch_idx], grid_size=grid_size, style=self.style)
            total_vis.append(block_vis)

        total_vis.append(f"torch.Tensor(shape={tuple(self.shape)}, dtype={self.dtype})")
        return "\n".join(total_vis)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, style: Optional[str] = None) -> "AttentionMask":
        res = cls(tensor)
        res.style = style
        return res