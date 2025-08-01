import copy
import math
import re
import numpy as np
import inspect
import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass, fields, is_dataclass

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn import init

zeros_ = lambda w: init.constant_(w, 0.0)
ones_ = lambda w: init.constant_(w, 1.0)
kaiming_uniform_ = lambda w: init.kaiming_uniform_(w, nonlinearity="relu")
trunc_normal_ = lambda w: init.trunc_normal_(w, std=0.02)
xavier_uniform_ = init.xavier_uniform_
xavier_normal_ = init.xavier_normal_


class ModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        class_fields = fields(self)

        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):           
            raise ValueError(
                f"{self.__class__.__name__} should not have more than one required field."
            )

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:]
        )
        if other_fields_are_none:
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            if first_field_iterator:
                for idx, element in enumerate(iterator):                  
                    if (
                        not isinstance(element, (list, tuple))
                        or not len(element) == 2
                        or not isinstance(element[0], str)
                    ):
                        if idx == 0:
                            self[class_fields[0].name] = first_field
                        else:                           
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance."
        )

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self):
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    cross_attentions: Optional[Tuple[torch.Tensor]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class Seq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class MBartConfig(object):
    model_type = "mbart"
    keys_to_ignore_at_inference = ["past_key_values"]  
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
    }

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        output_hidden_states=False,
        use_return_dict=True,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        _attn_implementation="eager",
        hidden_size=1024,
        use_parallel=False,
        parallel_step=2,
        is_export=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers   
        self.scale_embedding = scale_embedding
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.forced_eos_token_id = forced_eos_token_id
        self._attn_implementation = _attn_implementation
        self.use_parallel = use_parallel
        self.parallel_step = parallel_step
        self.is_export = is_export
        super().__init__()


@dataclass
class AttentionMaskConverter:
    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window=None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:        
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    @staticmethod
    def _make_causal_mask(
        input_ids_shape,
        dtype,
        device,
        past_key_values_length=0,
        sliding_window=None,
        is_export=False,
    ):
        bsz, tgt_len = input_ids_shape
        if is_export:         
            mask = torch.full(
                (tgt_len, tgt_len), torch.finfo(dtype).min, dtype=torch.float64, device=device
            )
        else:
            mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask = mask.masked_fill_(
            mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0
        )
        
        # paddle: return mask[None, None, :, :].expand(
        # paddle:     [bsz, 1, tgt_len, tgt_len + past_key_values_length]
        # paddle: )
        # Adjust for past_key_values_length. The original mask is for tgt_len x tgt_len.
        # It should be expanded to tgt_len x (tgt_len + past_key_values_length).
        # We need to add zeros for the past keys.
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        return mask.unsqueeze(0).unsqueeze(0).expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )
    
    # paddle: def to_4d_export(
    def to_4d_export(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        is_export=False,
    ):
        # paddle: input_shape = (attention_mask_2d.shape[0], query_length)
        input_shape = (attention_mask_2d.size(0), query_length)
        # paddle: expanded_attn_mask = self._expand_mask(
        # paddle:     attention_mask_2d, dtype, tgt_len=input_shape[-1]
        # paddle: )
        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        )
        # paddle: expanded_4d_mask = expanded_attn_mask
        expanded_4d_mask = expanded_attn_mask

        # paddle: return expanded_4d_mask
        return expanded_4d_mask

    # paddle: def to_4d(
    def to_4d(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        is_export=False,
    ):

        # paddle: input_shape = (attention_mask_2d.shape[0], query_length)
        input_shape = (attention_mask_2d.size(0), query_length)
        # paddle: causal_4d_mask = None
        causal_4d_mask = None
        # paddle: if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            # paddle: if key_value_length is None:
            if key_value_length is None:
                # paddle: raise ValueError(
                # paddle:     "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                # paddle: )
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            # paddle: past_key_values_length = key_value_length - query_length
            past_key_values_length = key_value_length - query_length

            # paddle: causal_4d_mask = self._make_causal_mask(
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device=attention_mask_2d.device,
                past_key_values_length=past_key_values_length,
                sliding_window=self.sliding_window,
                is_export=is_export,
            )
        # paddle: elif self.sliding_window is not None:
        elif self.sliding_window is not None:
            # paddle: raise NotImplementedError(
            # paddle:     "Sliding window is currently only implemented for causal masking"
            # paddle: )
            raise NotImplementedError(
                "Sliding window is currently only implemented for causal masking"
            )

        # paddle: expanded_attn_mask = self._expand_mask(
        # paddle:     attention_mask_2d, dtype, tgt_len=input_shape[-1]
        # paddle: )
        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        )

        # paddle: if causal_4d_mask is not None:
        if causal_4d_mask is not None:
            # paddle: if is_export:
            if is_export:
                # paddle: expanded_attn_mask = causal_4d_mask
                expanded_attn_mask = causal_4d_mask
                # paddle: return expanded_attn_mask
                return expanded_attn_mask
            # paddle: else:
            else:
                # paddle: expanded_attn_mask = causal_4d_mask.masked_fill_(
                # paddle:     expanded_attn_mask.cast(paddle.bool), paddle.finfo(dtype).min
                # paddle: )
                # The logic in paddle is a bit complex: it seems to combine causal mask and padding mask.
                # PyTorch behavior: adds the two masks together. `0` for positions to attend to, and a large negative number for masked positions.
                # A value of `0` in `expanded_attn_mask` corresponds to a padded position, which should be masked.
                # A value of `-inf` in `expanded_attn_mask` also corresponds to a masked position.
                expanded_attn_mask = causal_4d_mask + expanded_attn_mask

        # paddle: expanded_4d_mask = expanded_attn_mask
        expanded_4d_mask = expanded_attn_mask

        # paddle: return expanded_4d_mask
        return expanded_4d_mask

    # paddle: def _expand_mask(self, mask, dtype, tgt_len=None):
    def _expand_mask(self, mask, dtype, tgt_len=None):
        # paddle: bsz, src_len = mask.shape
        bsz, src_len = mask.size()
        # paddle: tgt_len = tgt_len if tgt_len is not None else src_len
        tgt_len = tgt_len if tgt_len is not None else src_len
        # paddle: expanded_mask = (
        # paddle:     mask[:, None, None, :].expand([bsz, 1, tgt_len, src_len]).cast(dtype)
        # paddle: )
        expanded_mask = (
            mask.unsqueeze(1).unsqueeze(2).expand(bsz, 1, tgt_len, src_len).to(dtype)
        )
        # paddle: inverted_mask = 1.0 - expanded_mask
        inverted_mask = 1.0 - expanded_mask
        # paddle: return inverted_mask.masked_fill_(
        # paddle:     inverted_mask.cast(paddle.bool), paddle.finfo(dtype).min
        # paddle: )
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )

# paddle: def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
    # paddle: return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)
    return AttentionMaskConverter(is_causal=False)._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


# paddle: def _prepare_4d_causal_attention_mask_export(
def _prepare_4d_causal_attention_mask_export(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    is_export=False,
):

    # paddle: attn_mask_converter = AttentionMaskConverter(
    # paddle:     is_causal=True, sliding_window=sliding_window
    # paddle: )
    attn_mask_converter = AttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window
    )
    # paddle: key_value_length = input_shape[-1] + past_key_values_length
    key_value_length = input_shape[-1] + past_key_values_length

    # paddle: shape = attention_mask.shape
    shape = attention_mask.size()
    # paddle: len_shape = len(shape)
    len_shape = len(shape)

    # paddle: attention_mask = attn_mask_converter.to_4d_export(
    attention_mask = attn_mask_converter.to_4d_export(
        attention_mask,
        input_shape[-1],
        key_value_length=key_value_length,
        dtype=inputs_embeds.dtype,
        is_export=is_export,
    )
    # paddle: return attention_mask
    return attention_mask


# paddle: def _prepare_4d_causal_attention_mask(
def _prepare_4d_causal_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    is_export=False,
):

    # paddle: attn_mask_converter = AttentionMaskConverter(
    # paddle:     is_causal=True, sliding_window=sliding_window
    # paddle: )
    attn_mask_converter = AttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window
    )
    # paddle: key_value_length = input_shape[-1] + past_key_values_length
    key_value_length = input_shape[-1] + past_key_values_length

    # paddle: shape = attention_mask.shape
    shape = attention_mask.size()
    # paddle: len_shape = len(shape)
    len_shape = len(shape)
    # paddle: if (attention_mask is not None) and (len_shape == 2):
    if (attention_mask is not None) and (len_shape == 2):
        # paddle: attention_mask = attn_mask_converter.to_4d(
        attention_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            is_export=is_export,
        )

        # paddle: return attention_mask
        return attention_mask
    # paddle: elif attention_mask is not None and len(attention_mask.shape) == 4:
    elif attention_mask is not None and len(attention_mask.size()) == 4:
        # paddle: expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        # paddle: if tuple(attention_mask.shape) != expected_shape:
        if tuple(attention_mask.size()) != expected_shape:
            # paddle: raise ValueError(
            # paddle:     f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            # paddle: )
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.size())}; expected: {expected_shape}."
            )
        # paddle: else:
        else:
            # paddle: inverted_mask = 1.0 - attention_mask
            inverted_mask = 1.0 - attention_mask
            # paddle: attention_mask = inverted_mask.masked_fill_(
            # paddle:     inverted_mask.to(paddle.bool), paddle.finfo(inputs_embeds.dtype).min
            # paddle: )
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    # paddle: else:
    else:
        # paddle: attention_mask = attn_mask_converter.to_causal_4d(
        # paddle:     input_shape[0],
        # paddle:     input_shape[-1],
        # paddle:     key_value_length,
        # paddle:     dtype=inputs_embeds.dtype,
        # paddle: )
        # to_causal_4d is not defined in the provided paddle code, assuming it's an alias for _make_causal_mask
        attention_mask = attn_mask_converter._make_causal_mask(
            (input_shape[0], input_shape[-1]),
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    # paddle: return attention_mask
    return attention_mask


# paddle: class MBartLearnedPositionalEmbedding(nn.Embedding):
class MBartLearnedPositionalEmbedding(nn.Embedding):
    # paddle: def __init__(self, num_embeddings, embedding_dim):
    def __init__(self, num_embeddings, embedding_dim):
        # paddle: self.offset = 2
        self.offset = 2
        # paddle: super().__init__(num_embeddings + self.offset, embedding_dim)
        super().__init__(num_embeddings + self.offset, embedding_dim)

    # paddle: def forward(self, input_ids, past_key_values_length=0):
    def forward(self, input_ids, past_key_values_length=0):
        # paddle: bsz, seq_len = input_ids.shape[:2]
        bsz, seq_len = input_ids.size()[:2]
        # paddle: positions = paddle.arange(
        # paddle:     past_key_values_length, past_key_values_length + seq_len, dtype=paddle.int64
        # paddle: ).expand([bsz, -1])
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)
        # paddle: return nn.Embedding.forward(self, positions + self.offset)
        return super().forward(positions + self.offset)


# paddle: class MBartPreTrainedModel(nn.Layer):
class MBartPreTrainedModel(nn.Module):
    # paddle: base_model_prefix = "model"
    base_model_prefix = "model"
    # paddle: supports_gradient_checkpointing = True
    supports_gradient_checkpointing = True
    # paddle: _no_split_modules = ["MBartDecoderLayer", "MBartAttention"]
    _no_split_modules = ["MBartDecoderLayer", "MBartAttention"]
    # paddle: _supports_flash_attn_2 = True
    _supports_flash_attn_2 = True

    # paddle: def __init__(self, config):
    def __init__(self, config):
        # paddle: super().__init__()
        super().__init__()
        # paddle: self.config = config
        self.config = config

    # paddle: def _initialize_weights(self, module):
    def _initialize_weights(self, module):
        # paddle: if getattr(module, "_is_hf_initialized", False):
        if getattr(module, "_is_hf_initialized", False):
            # paddle: return
            return
        # paddle: self._init_weights(module)
        self._init_weights(module)

    # paddle: def post_init(self):
    def post_init(self):
        # paddle: self.apply(self._initialize_weights)
        self.apply(self._initialize_weights)

    # paddle: def _init_weights(self, module):
    def _init_weights(self, module):
        # paddle: std = self.config.init_std
        std = self.config.init_std
        # paddle: normal_ = Normal(mean=0.0, std=std)
        normal_ = lambda w: init.normal_(w, mean=0.0, std=std)
        # paddle: if isinstance(module, nn.Linear):
        if isinstance(module, nn.Linear):
            # paddle: normal_(module.weight)
            normal_(module.weight)
            # paddle: if module.bias is not None:
            if module.bias is not None:
                # paddle: zeros_(module.bias)
                zeros_(module.bias)
        # paddle: elif isinstance(module, nn.Embedding):
        elif isinstance(module, nn.Embedding):
            # paddle: normal_(module.weight)
            normal_(module.weight)
            # paddle: if module._padding_idx is not None:
            if module.padding_idx is not None:
                # paddle: zeros_(module.weight[module._padding_idx])
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
    
    # paddle: @property
    # paddle: def dummy_inputs(self):
    @property
    def dummy_inputs(self):
        # paddle: pad_token = self.config.pad_token_id
        pad_token = self.config.pad_token_id
        # paddle: input_ids = paddle.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        # paddle: dummy_inputs = {
        # paddle:     "attention_mask": input_ids.ne(pad_token),
        # paddle:     "input_ids": input_ids,
        # paddle: }
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        # paddle: return dummy_inputs
        return dummy_inputs

# paddle: class MBartAttention(nn.Layer):
class MBartAttention(nn.Module):
    # paddle: def __init__(
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config=None,
    ):
        # paddle: super().__init__()
        super().__init__()
        # paddle: self.embed_dim = embed_dim
        self.embed_dim = embed_dim
        # paddle: self.num_heads = num_heads
        self.num_heads = num_heads
        # paddle: self.dropout = dropout
        self.dropout = dropout
        # paddle: self.head_dim = embed_dim // num_heads
        self.head_dim = embed_dim // num_heads
        # paddle: self.config = config
        self.config = config

        # paddle: if (self.head_dim * num_heads) != self.embed_dim:
        if (self.head_dim * num_heads) != self.embed_dim:
            # paddle: raise ValueError(
            # paddle:     f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
            # paddle:     f" and `num_heads`: {num_heads})."
            # paddle: )
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # paddle: self.scaling = self.head_dim**-0.5
        self.scaling = self.head_dim**-0.5
        # paddle: self.is_decoder = is_decoder
        self.is_decoder = is_decoder
        # paddle: self.is_causal = is_causal
        self.is_causal = is_causal

        # paddle: self.k_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # paddle: self.v_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # paddle: self.q_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # paddle: self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # paddle: def _shape(self, tensor, seq_len, bsz):
    def _shape(self, tensor, seq_len, bsz):
        # paddle: return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose(
        # paddle:     [0, 2, 1, 3]
        # paddle: )
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    # paddle: def forward(
    def forward(
        self,
        hidden_states,
        key_value_states=None,
        past_key_value=None,
        attention_mask=None,
        layer_head_mask=None,
        output_attentions=False,
    ):

        # paddle: is_cross_attention = key_value_states is not None
        is_cross_attention = key_value_states is not None

        # paddle: bsz, tgt_len, _ = paddle.shape(hidden_states)
        bsz, tgt_len, _ = hidden_states.size()
        # paddle: query_states = self.q_proj(hidden_states) * self.scaling
        query_states = self.q_proj(hidden_states) * self.scaling
        # paddle: if (
        # paddle:     is_cross_attention
        # paddle:     and past_key_value is not None
        # paddle:     and past_key_value[0].shape[2] == key_value_states.shape[1]
        # paddle: ):
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].size(2) == key_value_states.size(1)
        ):
            # paddle: key_states = past_key_value[0]
            key_states = past_key_value[0]
            # paddle: value_states = past_key_value[1]
            value_states = past_key_value[1]
        # paddle: elif is_cross_attention:
        elif is_cross_attention:
            # paddle: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            # paddle: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        # paddle: elif past_key_value is not None:
        elif past_key_value is not None:
            # paddle: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            # paddle: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            # paddle: key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            # paddle: value_states = paddle.concat([past_key_value[1], value_states], axis=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        # paddle: else:
        else:
            # paddle: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            # paddle: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # paddle: if self.is_decoder:
        if self.is_decoder:
            # paddle: past_key_value = (key_states, value_states)
            past_key_value = (key_states, value_states)

        # paddle: proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # paddle: query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        # paddle: key_states = key_states.reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        # paddle: value_states = value_states.reshape(proj_shape)
        value_states = value_states.reshape(proj_shape)

        # paddle: src_len = key_states.shape[1]
        src_len = key_states.size(1)
        # paddle: attn_weights = paddle.bmm(query_states, key_states.transpose([0, 2, 1]))
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        # paddle: if attention_mask is not None:
        if attention_mask is not None:
            # paddle: attn_weights = (
            # paddle:     attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            # paddle:     + attention_mask
            # paddle: )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            # paddle: attn_weights = attn_weights.reshape(
            # paddle:     [bsz * self.num_heads, tgt_len, src_len]
            # paddle: )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        # paddle: attn_weights = nn.functional.softmax(attn_weights, axis=-1)
        attn_weights = F.softmax(attn_weights, dim=-1)
        # paddle: if layer_head_mask is not None:
        if layer_head_mask is not None:
            # paddle: if tuple(layer_head_mask.shape) != (self.num_heads,):
            if tuple(layer_head_mask.size()) != (self.num_heads,):
                # paddle: raise ValueError(
                # paddle:     f"Head mask for a single layer should be of shape {(self.num_heads,)}, but is"
                # paddle:     f" {layer_head_mask.shape}"
                # paddle: )
                raise ValueError(
                    f"Head mask for a single layer should be of shape {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            # paddle: attn_weights = layer_head_mask.reshape(
            # paddle:     [1, -1, 1, 1]
            # paddle: ) * attn_weights.reshape([bsz, self.num_heads, tgt_len, src_len])
            attn_weights = layer_head_mask.view(
                1, -1, 1, 1
            ) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            # paddle: attn_weights = attn_weights.reshape(
            # paddle:     [bsz * self.num_heads, tgt_len, src_len]
            # paddle: )
            attn_weights = attn_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        # paddle: if output_attentions:
        if output_attentions:
            # paddle: attn_weights_reshaped = attn_weights.reshape(
            # paddle:     [bsz, self.num_heads, tgt_len, src_len]
            # paddle: )
            attn_weights_reshaped = attn_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            # paddle: attn_weights = attn_weights_reshaped.reshape(
            # paddle:     [bsz * self.num_heads, tgt_len, src_len]
            # paddle: )
            attn_weights = attn_weights_reshaped.view(
                bsz * self.num_heads, tgt_len, src_len
            )
        # paddle: else:
        else:
            # paddle: attn_weights_reshaped = None
            attn_weights_reshaped = None
        # paddle: attn_probs = nn.functional.dropout(
        # paddle:     attn_weights, p=self.dropout, training=self.training
        # paddle: )
        attn_probs = F.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        # paddle: attn_output = paddle.bmm(attn_probs, value_states)
        attn_output = torch.bmm(attn_probs, value_states)

        # paddle: attn_output = attn_output.reshape([bsz, self.num_heads, tgt_len, self.head_dim])
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        # paddle: attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.permute(0, 2, 1, 3)

        # paddle: attn_output = attn_output.reshape([bsz, tgt_len, self.embed_dim])
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        # paddle: attn_output = self.out_proj(attn_output)
        attn_output = self.out_proj(attn_output)
        # paddle: return attn_output, attn_weights_reshaped, past_key_value
        return attn_output, attn_weights_reshaped, past_key_value


# paddle: MBART_ATTENTION_CLASSES = {
# paddle:     "eager": MBartAttention,
# paddle: }
MBART_ATTENTION_CLASSES = {
    "eager": MBartAttention,
}


# paddle: class MBartDecoderLayer(nn.Layer):
class MBartDecoderLayer(nn.Module):
    # paddle: def __init__(self, config):
    def __init__(self, config):
        # paddle: super().__init__()
        super().__init__()
        # paddle: self.embed_dim = config.d_model
        self.embed_dim = config.d_model
        # paddle: self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
        self.self_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # paddle: self.is_export = config.is_export
        self.is_export = config.is_export
        # paddle: self.dropout = config.dropout
        self.dropout = config.dropout
        # paddle: self.activation_fn = F.gelu
        self.activation_fn = F.gelu
        # paddle: self.activation_dropout = config.activation_dropout
        self.activation_dropout = config.activation_dropout

        # paddle: self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # paddle: self.encoder_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
        self.encoder_attn = MBART_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # paddle: self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # paddle: self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # paddle: self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # paddle: self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # paddle: def forward(
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> torch.Tensor:

        # paddle: residual = hidden_states
        residual = hidden_states
        # paddle: hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # paddle: self_attn_past_key_value = (
        # paddle:     past_key_value[:2] if past_key_value is not None else None
        # paddle: )
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )

        # paddle: hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # paddle: hidden_states = nn.functional.dropout(
        # paddle:     hidden_states, p=self.dropout, training=self.training
        # paddle: )
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        # paddle: hidden_states = residual + hidden_states
        hidden_states = residual + hidden_states

        # paddle: cross_attn_present_key_value = None
        cross_attn_present_key_value = None
        # paddle: cross_attn_weights = None
        cross_attn_weights = None
        # paddle: if encoder_hidden_states is not None:
        if encoder_hidden_states is not None:
            # paddle: residual = hidden_states
            residual = hidden_states
            # paddle: hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # paddle: cross_attn_past_key_value = (
            # paddle:     past_key_value[-2:] if past_key_value is not None else None
            # paddle: )
            cross_attn_past_key_value = (
                past_key_value[-2:] if past_key_value is not None else None
            )
            # paddle: hidden_states, cross_attn_weights, cross_attn_present_key_value = (
            hidden_states, cross_attn_weights, cross_attn_present_key_value = (
                # paddle: self.encoder_attn(
                self.encoder_attn(
                    hidden_states=hidden_states,
                    key_value_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=cross_attn_past_key_value,
                    output_attentions=output_attentions,
                )
            )
            # paddle: hidden_states = nn.functional.dropout(
            # paddle:     hidden_states, p=self.dropout, training=self.training
            # paddle: )
            hidden_states = F.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            # paddle: hidden_states = residual + hidden_states
            hidden_states = residual + hidden_states

            # paddle: present_key_value = present_key_value + cross_attn_present_key_value
            present_key_value = present_key_value + cross_attn_present_key_value

        # paddle: residual = hidden_states
        residual = hidden_states
        # paddle: hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        # paddle: hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # paddle: hidden_states = nn.functional.dropout(
        # paddle:     hidden_states, p=self.activation_dropout, training=self.training
        # paddle: )
        hidden_states = F.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        # paddle: hidden_states = self.fc2(hidden_states)
        hidden_states = self.fc2(hidden_states)
        # paddle: hidden_states = nn.functional.dropout(
        # paddle:     hidden_states, p=self.dropout, training=self.training
        # paddle: )
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        # paddle: hidden_states = residual + hidden_states
        hidden_states = residual + hidden_states
        # paddle: outputs = (hidden_states,)
        outputs = (hidden_states,)

        # paddle: if output_attentions:
        if output_attentions:
            # paddle: outputs += (self_attn_weights, cross_attn_weights)
            outputs += (self_attn_weights, cross_attn_weights)

        # paddle: if self.is_export:
        if self.is_export:
            # paddle: outputs += (present_key_value,)
            outputs += (present_key_value,)
        # paddle: else:
        else:
            # paddle: if use_cache:
            if use_cache:
                # paddle: outputs += (present_key_value,)
                outputs += (present_key_value,)
        # paddle: return outputs
        return outputs


# paddle: class MBartForCausalLM(MBartPreTrainedModel):
class MBartForCausalLM(MBartPreTrainedModel):
    # paddle: _tied_weights_keys = ["lm_head.weight"]
    _tied_weights_keys = ["lm_head.weight"]

    # paddle: def __init__(self, config):
    def __init__(self, config):
        # paddle: config = copy.deepcopy(config)
        config = copy.deepcopy(config)
        # paddle: config.is_decoder = True
        config.is_decoder = True
        # paddle: config.is_encoder_decoder = False
        config.is_encoder_decoder = False
        # paddle: super().__init__(config)
        super().__init__(config)
        # paddle: self.model = MBartDecoderWrapper(config)
        self.model = MBartDecoderWrapper(config)
        # paddle: self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # paddle: self.post_init()
        self.post_init()

    # paddle: def get_input_embeddings(self):
    def get_input_embeddings(self):
        # paddle: return self.model.decoder.embed_tokens
        return self.model.decoder.embed_tokens

    # paddle: def set_input_embeddings(self, value):
    def set_input_embeddings(self, value):
        # paddle: self.model.decoder.embed_tokens = value
        self.model.decoder.embed_tokens = value

    # paddle: def get_output_embeddings(self):
    def get_output_embeddings(self):
        # paddle: return self.lm_head
        return self.lm_head

    # paddle: def set_output_embeddings(self, new_embeddings):
    def set_output_embeddings(self, new_embeddings):
        # paddle: self.lm_head = new_embeddings
        self.lm_head = new_embeddings

    # paddle: def set_decoder(self, decoder):
    def set_decoder(self, decoder):
        # paddle: self.model.decoder = decoder
        self.model.decoder = decoder

    # paddle: def get_decoder(self):
    def get_decoder(self):
        # paddle: return self.model.decoder
        return self.model.decoder

    # paddle: def forward(
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # paddle: output_attentions = (
        # paddle:     output_attentions
        # paddle:     if output_attentions is not None
        # paddle:     else self.config.output_attentions
        # paddle: )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # paddle: output_hidden_states = (
        # paddle:     output_hidden_states
        # paddle:     if output_hidden_states is not None
        # paddle:     else self.config.output_hidden_states
        # paddle: )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # paddle: return_dict = (
        # paddle:     return_dict if return_dict is not None else self.config.use_return_dict
        # paddle: )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # paddle: outputs = self.model.decoder(
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # paddle: logits = self.lm_head(outputs[0])
        # In PyTorch, if return_dict=True, output is an object.
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        # paddle: loss = None
        loss = None
        # paddle: if labels is not None:
        if labels is not None:
            # paddle: labels = labels
            labels = labels
            # paddle: loss_fct = CrossEntropyLoss()
            loss_fct = CrossEntropyLoss()
            # paddle: loss = loss_fct(
            # paddle:     logits.reshape([-1, self.config.vocab_size]), labels.reshape([-1])
            # paddle: )
            loss = loss_fct(
                logits.view(-1, self.config.vocab_size), labels.view(-1)
            )

        # paddle: if not return_dict:
        if not return_dict:
            # paddle: output = (logits,) + outputs[1:]
            output = (logits,) + outputs[1:]
            # paddle: return (loss,) + output if loss is not None else output
            return (loss,) + output if loss is not None else output

        # paddle: return CausalLMOutputWithCrossAttentions(
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    # paddle: def prepare_inputs_for_generation(
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # paddle: if attention_mask is None:
        # paddle:     attention_mask = input_ids.new_ones(input_ids.shape)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # paddle: if past_key_values:
        if past_key_values:
            # paddle: past_length = past_key_values[0][0].shape[2]
            past_length = past_key_values[0][0].size(2)

            # paddle: if input_ids.shape[1] > past_length:
            if input_ids.size(1) > past_length:
                # paddle: remove_prefix_length = past_length
                remove_prefix_length = past_length
            # paddle: else:
            else:
                # paddle: remove_prefix_length = input_ids.shape[1] - 1
                remove_prefix_length = input_ids.size(1) - 1
            
            # paddle: input_ids = input_ids[:, remove_prefix_length:]
            input_ids = input_ids[:, remove_prefix_length:]

        # paddle: return {
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # paddle: @staticmethod
    # paddle: def _reorder_cache(past_key_values, beam_idx):
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # paddle: reordered_past = ()
        reordered_past = ()
        # paddle: for layer_past in past_key_values:
        for layer_past in past_key_values:
            # paddle: reordered_past += (
            # paddle:     tuple(
            # paddle:         past_state.index_select(0, beam_idx) for past_state in layer_past
            # paddle:     ),
            # paddle: )
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx) for past_state in layer_past
                ),
            )
        # paddle: return reordered_past
        return reordered_past


# paddle: class myLayerNorm(nn.LayerNorm):
class myLayerNorm(nn.LayerNorm):
    # paddle: def __init__(
    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        drop_block=None,
    ):
        # paddle: super(nn.LayerNorm, self).__init__()
        # PyTorch LayerNorm takes normalized_shape
        super(myLayerNorm, self).__init__(num_channels, eps=eps, elementwise_affine=affine)
        # paddle: self._epsilon = eps
        # self.eps is already set in super().__init__
        # paddle: self.num_channels = num_channels
        self.num_channels = num_channels
        # paddle: if affine:
        # paddle:     self.weight = paddle.create_parameter([num_channels], dtype="float32")
        # paddle:     self.bias = paddle.create_parameter([num_channels], dtype="float32")
        # paddle:     ones_(self.weight)
        # paddle:     zeros_(self.bias)
        # PyTorch nn.LayerNorm handles this automatically if elementwise_affine=True

    # paddle: def forward(self, x):
    def forward(self, x):
        # paddle: x = F.layer_norm(
        # paddle:     x,
        # paddle:     self.num_channels,
        # paddle:     weight=self.weight,
        # paddle:     bias=self.bias,
        # paddle:     epsilon=self._epsilon,
        # paddle: )
        # Using the parent's forward method which does the same thing
        x = super().forward(x)
        # paddle: return x
        return x


# paddle: class MBartDecoder(MBartPreTrainedModel):
class MBartDecoder(MBartPreTrainedModel):
    # paddle: def __init__(self, config, embed_tokens=None):
    def __init__(self, config, embed_tokens=None):
        # paddle: super().__init__(config)
        super().__init__(config)
        # paddle: self.dropout = config.dropout
        self.dropout = config.dropout
        # paddle: self.layerdrop = config.decoder_layerdrop
        self.layerdrop = config.decoder_layerdrop
        # paddle: self.padding_idx = config.pad_token_id
        self.padding_idx = config.pad_token_id
        # paddle: self.max_target_positions = config.max_position_embeddings
        self.max_target_positions = config.max_position_embeddings
        # paddle: self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # paddle: self.embed_tokens = nn.Embedding(
        # paddle:     config.vocab_size, config.d_model, self.padding_idx
        # paddle: )
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model, self.padding_idx
        )

        # paddle: if embed_tokens is not None:
        if embed_tokens is not None:
            # paddle: self.embed_tokens.weight = embed_tokens.weight
            self.embed_tokens.weight = embed_tokens.weight

        # paddle: self.embed_positions = MBartLearnedPositionalEmbedding(
        # paddle:     config.max_position_embeddings,
        # paddle:     config.d_model,
        # paddle: )
        self.embed_positions = MBartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        # paddle: self.layers = nn.LayerList(
        # paddle:     [MBartDecoderLayer(config) for _ in range(config.decoder_layers)]
        # paddle: )
        self.layers = nn.ModuleList(
            [MBartDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        # paddle: self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        # paddle: self.layernorm_embedding = myLayerNorm(config.d_model, affine=True)
        self.layernorm_embedding = myLayerNorm(config.d_model, affine=True)
        # paddle: self.layer_norm = nn.LayerNorm(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)

        # paddle: self.gradient_checkpointing = False
        self.gradient_checkpointing = False
        # paddle: self.post_init()
        self.post_init()
        # paddle: self.is_export = config.is_export
        self.is_export = config.is_export

    # paddle: def get_input_embeddings(self):
    def get_input_embeddings(self):
        # paddle: return self.embed_tokens
        return self.embed_tokens

    # paddle: def set_input_embeddings(self, value):
    def set_input_embeddings(self, value):
        # paddle: self.embed_tokens = value
        self.embed_tokens = value

    # paddle: def forward(
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # paddle: output_attentions = (
        # paddle:     output_attentions
        # paddle:     if output_attentions is not None
        # paddle:     else self.config.output_attentions
        # paddle: )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # paddle: output_hidden_states = (
        # paddle:     output_hidden_states
        # paddle:     if output_hidden_states is not None
        # paddle:     else self.config.output_hidden_states
        # paddle: )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # paddle: use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # paddle: return_dict = (
        # paddle:     return_dict if return_dict is not None else self.config.use_return_dict
        # paddle: )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # paddle: if input_ids is not None and inputs_embeds is not None:
        if input_ids is not None and inputs_embeds is not None:
            # paddle: raise ValueError(
            # paddle:     "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            # paddle: )
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        # paddle: elif input_ids is not None:
        elif input_ids is not None:
            # paddle: input = input_ids
            input = input_ids
            # paddle: input_shape = input.shape
            input_shape = input.size()
            # paddle: input_ids = input_ids.reshape([-1, input_shape[-1]])
            input_ids = input_ids.view(-1, input_shape[-1])
        # paddle: elif inputs_embeds is not None:
        elif inputs_embeds is not None:
            # paddle: input_shape = inputs_embeds.shape[:-1]
            input_shape = inputs_embeds.size()[:-1]
            # paddle: input = inputs_embeds[:, :, -1]
            input = inputs_embeds[:, :, -1]
        # paddle: else:
        else:
            # paddle: raise ValueError(
            # paddle:     "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            # paddle: )
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # paddle: past_key_values_length = (
        # paddle:     past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # paddle: )
        past_key_values_length = (
            past_key_values[0][0].size(2) if past_key_values is not None else 0
        )

        # paddle: if inputs_embeds is None:
        if inputs_embeds is None:
            # paddle: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        
        # paddle: if self._use_flash_attention_2:
        if self._use_flash_attention_2:
            # paddle: attention_mask = (
            # paddle:     attention_mask
            # paddle:     if (attention_mask is not None and 0 in attention_mask)
            # paddle:     else None
            # paddle: )
            # PyTorch flash attention typically handles padding mask internally, but setting it to None if all are 1 is a common pattern.
            attention_mask = (
                attention_mask
                if (attention_mask is not None and (attention_mask == 0).any())
                else None
            )
        # paddle: else:
        else:
            # paddle: attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
                is_export=self.is_export,
            )

        # paddle: if encoder_hidden_states is not None and encoder_attention_mask is not None:
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # paddle: if self._use_flash_attention_2:
            if self._use_flash_attention_2:
                # paddle: encoder_attention_mask = (
                # paddle:     encoder_attention_mask if 0 in encoder_attention_mask else None
                # paddle: )
                encoder_attention_mask = (
                    encoder_attention_mask if (encoder_attention_mask == 0).any() else None
                )
            # paddle: else:
            else:
                # paddle: encoder_attention_mask = _prepare_4d_attention_mask(
                # paddle:     encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                # paddle: )
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # paddle: positions = self.embed_positions(input, past_key_values_length)
        positions = self.embed_positions(input_ids, past_key_values_length)

        # paddle: hidden_states = inputs_embeds + positions
        hidden_states = inputs_embeds + positions
        # paddle: hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.layernorm_embedding(hidden_states)

        # paddle: hidden_states = nn.functional.dropout(
        # paddle:     hidden_states, p=self.dropout, training=self.training
        # paddle: )
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # paddle: if self.gradient_checkpointing and self.training:
        if self.gradient_checkpointing and self.training:
            # paddle: if use_cache:
            if use_cache:
                # paddle: print(
                # paddle:     "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                # paddle: )
                warnings.warn(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                # paddle: use_cache = False
                use_cache = False

        # paddle: all_hidden_states = () if output_hidden_states else None
        all_hidden_states = () if output_hidden_states else None
        # paddle: all_self_attns = () if output_attentions else None
        all_self_attns = () if output_attentions else None
        # paddle: all_cross_attentions = (
        # paddle:     () if (output_attentions and encoder_hidden_states is not None) else None
        # paddle: )
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        # paddle: next_decoder_cache = () if use_cache else None
        next_decoder_cache = () if use_cache else None

        # paddle: for attn_mask, mask_name in zip(
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            # paddle: if attn_mask is not None:
            if attn_mask is not None:
                # paddle: if attn_mask.shape[0] != len(self.layers):
                if attn_mask.size(0) != len(self.layers):
                    # paddle: raise ValueError(
                    # paddle:     f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    # paddle:     f" {attn_mask.shape[0]}."
                    # paddle: )
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size(0)}."
                    )

        # paddle: for idx, decoder_layer in enumerate(self.layers):
        for idx, decoder_layer in enumerate(self.layers):
            # paddle: if output_hidden_states:
            if output_hidden_states:
                # paddle: all_hidden_states += (hidden_states,)
                all_hidden_states += (hidden_states,)
            # paddle: if self.training:
            if self.training:
                # paddle: dropout_probability = paddle.rand([])
                dropout_probability = torch.rand(1)
                # paddle: if dropout_probability < self.layerdrop:
                if dropout_probability < self.layerdrop:
                    # paddle: continue
                    continue

            # paddle: past_key_value = (
            # paddle:     past_key_values[idx] if past_key_values is not None else None
            # paddle: )
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # paddle: if self.gradient_checkpointing and self.training:
            if self.gradient_checkpointing and self.training:
                # paddle: layer_outputs = self._gradient_checkpointing_func(
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions=output_attentions, use_cache=use_cache)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None, # past_key_value is passed implicitly
                )
            # paddle: else:
            else:
                # paddle: layer_outputs = decoder_layer(
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            # paddle: hidden_states = layer_outputs[0]
            hidden_states = layer_outputs[0]

            # paddle: if use_cache:
            if use_cache:
                # paddle: next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            # paddle: if output_attentions:
            if output_attentions:
                # paddle: all_self_attns += (layer_outputs[1],)
                all_self_attns += (layer_outputs[1],)

                # paddle: if encoder_hidden_states is not None:
                if encoder_hidden_states is not None:
                    # paddle: all_cross_attentions += (layer_outputs[2],)
                    all_cross_attentions += (layer_outputs[2],)

        # paddle: hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # paddle: if output_hidden_states:
        if output_hidden_states:
            # paddle: all_hidden_states += (hidden_states,)
            all_hidden_states += (hidden_states,)

        # paddle: next_cache = next_decoder_cache if use_cache else None
        next_cache = next_decoder_cache if use_cache else None
        # paddle: if not return_dict:
        if not return_dict:
            # paddle: return tuple(
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # paddle: return BaseModelOutputWithPastAndCrossAttentions(
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# paddle: class MBartDecoderWrapper(MBartPreTrainedModel):
class MBartDecoderWrapper(MBartPreTrainedModel):
    # paddle: def __init__(self, config):
    def __init__(self, config):
        # paddle: super().__init__(config)
        super().__init__(config)
        # paddle: self.decoder = MBartDecoder(config)
        self.decoder = MBartDecoder(config)

    # paddle: def forward(self, *args, **kwargs):
    def forward(self, *args, **kwargs):
        # paddle: return self.decoder(*args, **kwargs)
        return self.decoder(*args, **kwargs)

# paddle: def _in_projection(
def _in_projection(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    b_q: Optional[torch.Tensor] = None,
    b_k: Optional[torch.Tensor] = None,
    b_v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # paddle: Eq, Ek, Ev = q.shape[-1], k.shape[-1], v.shape[-1]
    Eq, Ek, Ev = q.size(-1), k.size(-1), v.size(-1)
    # paddle: assert w_q.shape == (Eq, Eq,), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.shape}"
    assert w_q.size() == (Eq, Eq,), f"expecting query weights shape of {(Eq, Eq)}, but got {w_q.size()}"
    # paddle: assert w_k.shape == (Eq, Ek,), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.shape}"
    assert w_k.size() == (Eq, Ek,), f"expecting key weights shape of {(Eq, Ek)}, but got {w_k.size()}"
    # paddle: assert w_v.shape == (Eq, Ev,), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.shape}"
    assert w_v.size() == (Eq, Ev,), f"expecting value weights shape of {(Eq, Ev)}, but got {w_v.size()}"
    # paddle: assert b_q is None or b_q.shape == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.shape}"
    assert b_q is None or b_q.size() == (Eq,), f"expecting query bias shape of {(Eq,)}, but got {b_q.size()}"
    # paddle: assert b_k is None or b_k.shape == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.shape}"
    assert b_k is None or b_k.size() == (Eq,), f"expecting key bias shape of {(Eq,)}, but got {b_k.size()}"
    # paddle: assert b_v is None or b_v.shape == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.shape}"
    assert b_v is None or b_v.size() == (Eq,), f"expecting value bias shape of {(Eq,)}, but got {b_v.size()}"
    # paddle: return linear(q, w_q.T, b_q), linear(k, w_k.T, b_k), linear(v, w_v.T, b_v)
    return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


# paddle: def _scaled_dot_product_attention(
def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # paddle: B, Nt, E = q.shape
    B, Nt, E = q.size()
    # paddle: q = q / math.sqrt(E)
    q = q / math.sqrt(E)
    # paddle: attn = paddle.bmm(q, k.transpose([0, 2, 1]))
    attn = torch.bmm(q, k.transpose(1, 2))
    # paddle: if attn_mask is not None:
    if attn_mask is not None:
        # paddle: attn += attn_mask
        attn += attn_mask
    # paddle: attn = F.softmax(attn, axis=-1)
    attn = F.softmax(attn, dim=-1)
    # paddle: if dropout_p > 0.0:
    if dropout_p > 0.0:
        # paddle: attn = F.dropout(attn, p=dropout_p)
        attn = F.dropout(attn, p=dropout_p)
    # paddle: output = paddle.bmm(attn, v)
    output = torch.bmm(attn, v)
    # paddle: return output, attn
    return output, attn


# paddle: def linear(x, w, b, is_transpose):
def linear(x, w, b, is_transpose):
    # paddle: if b is not None:
    # paddle:     return paddle.matmul(x, w, transpose_y=is_transpose) + b
    # paddle: else:
    # paddle:     return paddle.matmul(x, w, transpose_y=is_transpose)
    # PyTorch's F.linear expects w to be (out_features, in_features)
    # The paddle code uses transpose_y, so we transpose w for PyTorch.
    weight = w.t() if not is_transpose else w
    return F.linear(x, weight, b)


# paddle: def _in_projection_packed(
def _in_projection_packed(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w: Tensor,
    b: Optional[Tensor] = None,
    is_export=False,
) -> List[Tensor]:

    # paddle: E = paddle.shape(q)[-1]
    E = q.size(-1)
    # paddle: if k is v:
    if torch.equal(k, v):
        # paddle: if q is k:
        if torch.equal(q,k):
            # paddle: proj = linear(q, w, b, is_transpose=True)
            proj = F.linear(q, w, b) # PyTorch linear's weight is already (out, in)
            # paddle: if is_export:
            # The export logic with unflatten/transpose is specific. 
            # Replicating it requires careful tensor manipulation.
            if is_export:
                # paddle: B, D, L = paddle.shape(proj)
                # paddle: proj = proj.reshape([B, D, 3, E])
                # paddle: proj = (
                # paddle:     proj.unsqueeze(0)
                # paddle:     .transpose([3, 1, 2, 0, 4])
                # paddle:     .squeeze(-2)
                # paddle:     .contiguous()
                # paddle: )
                proj = proj.unflatten(-1, (3, E)).unsqueeze(0).permute(3, 1, 2, 0, 4).squeeze(-2).contiguous()
            # paddle: else:
            else:
                # paddle: proj = (
                # paddle:     proj.unflatten(-1, (3, E))
                # paddle:     .unsqueeze(0)
                # paddle:     .transpose([3, 1, 2, 0, 4])
                # paddle:     .squeeze(-2)
                # paddle:     .contiguous()
                # paddle: )
                proj = proj.unflatten(-1, (3, E)).unsqueeze(0).permute(3, 1, 2, 0, 4).squeeze(-2).contiguous()
            # paddle: return proj[0], proj[1], proj[2]
            return proj[0], proj[1], proj[2]
    # paddle: else:
    else:
        # paddle: w_q, w_k, w_v = w.chunk(3)
        w_q, w_k, w_v = w.chunk(3)
        # paddle: if b is None:
        if b is None:
            # paddle: b_q = b_k = b_v = None
            b_q = b_k = b_v = None
        # paddle: else:
        else:
            # paddle: b_q, b_k, b_v = b.chunk(3)
            b_q, b_k, b_v = b.chunk(3)
        # paddle: return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)


# paddle: def multi_head_attention_forward(
def multi_head_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: torch.Tensor,
    in_proj_bias: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: Optional[torch.Tensor],
    training: bool = True,
    key_padding_mask: Optional[torch.Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[torch.Tensor] = None,
    k_proj_weight: Optional[torch.Tensor] = None,
    v_proj_weight: Optional[torch.Tensor] = None,
    static_k: Optional[torch.Tensor] = None,
    static_v: Optional[torch.Tensor] = None,
    is_export=False,
):

    # paddle: tgt_len, bsz, embed_dim = query.shape
    tgt_len, bsz, embed_dim = query.shape
    # paddle: src_len, _, _ = key.shape
    src_len, _, _ = key.shape

    # paddle: if isinstance(embed_dim, paddle.Tensor):
    # paddle:     head_dim = embed_dim.div(num_heads, rounding_mode="trunc")
    # paddle: else:
    # paddle:     head_dim = embed_dim // num_heads
    # In PyTorch, shape values are ints
    head_dim = embed_dim // num_heads
    # paddle: q, k, v = _in_projection_packed(
    # paddle:     query, key, value, in_proj_weight, in_proj_bias, is_export
    # paddle: )
    q, k, v = _in_projection_packed(
        query, key, value, in_proj_weight, in_proj_bias, is_export
    )

    # paddle: if key_padding_mask is not None and key_padding_mask.dtype == paddle.uint8:
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        # paddle: warnings.warn(
        # paddle:     "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        # paddle: )
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        # paddle: key_padding_mask = key_padding_mask.to(paddle.bool)
        key_padding_mask = key_padding_mask.to(torch.bool)
    
    # This logic seems incorrect in the original code, as it's always False.
    # paddle: if bias_k is not None and bias_v is not None:  # False
    if bias_k is not None and bias_v is not None:
        # paddle: assert static_k is None, "bias cannot be added to static key."
        assert static_k is None, "bias cannot be added to static key."
        # paddle: assert static_v is None, "bias cannot be added to static value."
        assert static_v is None, "bias cannot be added to static value."
        # paddle: k = paddle.concat([k, bias_k.repeat(1, bsz, 1)])
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        # paddle: v = paddle.concat([v, bias_v.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
    # paddle: else:
    else:
        # paddle: assert bias_k is None
        assert bias_k is None
        # paddle: assert bias_v is None
        assert bias_v is None

    # paddle: q = q.reshape([tgt_len, bsz * num_heads, head_dim]).transpose([1, 0, 2])
    q = q.reshape(tgt_len, bsz * num_heads, head_dim).permute(1, 0, 2)
    # paddle: if static_k is None:  # True
    if static_k is None:
        # paddle: k = k.reshape([k.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
        k = k.reshape(k.shape[0], bsz * num_heads, head_dim).permute(1, 0, 2)
    # paddle: else:
    else:
        # paddle: assert (
        # paddle:     static_k.shape[0] == bsz * num_heads
        # paddle: ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.shape[0]}"
        assert (
            static_k.size(0) == bsz * num_heads
        ), f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        # paddle: assert (
        # paddle:     static_k.shape[2] == head_dim
        # paddle: ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.shape[2]}"
        assert (
            static_k.size(2) == head_dim
        ), f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        # paddle: k = static_k
        k = static_k
    # paddle: if static_v is None:  # True
    if static_v is None:
        # paddle: v = v.reshape([v.shape[0], bsz * num_heads, head_dim]).transpose([1, 0, 2])
        v = v.reshape(v.shape[0], bsz * num_heads, head_dim).permute(1, 0, 2)
    # paddle: else:
    else:
        # paddle: assert (
        # paddle:     static_v.shape[0] == bsz * num_heads
        # paddle: ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.shape[0]}"
        assert (
            static_v.size(0) == bsz * num_heads
        ), f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        # paddle: assert (
        # paddle:     static_v.shape[2] == head_dim
        # paddle: ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.shape[2]}"
        assert (
            static_v.size(2) == head_dim
        ), f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        # paddle: v = static_v
        v = static_v

    # paddle: src_len = k.shape[1]
    src_len = k.size(1)

    # paddle: if not training:
    if not training:
        # paddle: dropout_p = 0.0
        dropout_p = 0.0

    # paddle: attn_output, attn_output_weights = _scaled_dot_product_attention(
    # paddle:     q, k, v, attn_mask, dropout_p
    # paddle: )
    attn_output, attn_output_weights = _scaled_dot_product_attention(
        q, k, v, attn_mask, dropout_p
    )

    # paddle: attn_output = attn_output.transpose([1, 0, 2]).reshape([tgt_len, bsz, embed_dim])
    attn_output = attn_output.permute(1, 0, 2).reshape(tgt_len, bsz, embed_dim)
    # paddle: attn_output = linear(
    # paddle:     attn_output, out_proj_weight, out_proj_bias, is_transpose=False
    # paddle: )
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    # paddle: if need_weights:
    if need_weights:
        # paddle: attn_output_weights = attn_output_weights.reshape(
        # paddle:     [bsz, num_heads, tgt_len, src_len]
        # paddle: )
        attn_output_weights = attn_output_weights.view(
            bsz, num_heads, tgt_len, src_len
        )
        # paddle: return attn_output, attn_output_weights.sum(axis=1) / num_heads
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    # paddle: else:
    else:
        # paddle: return attn_output, None
        return attn_output, None


# paddle: class MyMultiheadAttention(nn.Layer):
class MyMultiheadAttention(nn.Module):
    # paddle: __constants__ = ["batch_first"]
    __constants__ = ["batch_first"]
    # paddle: bias_k: Optional[paddle.Tensor]
    bias_k: Optional[torch.Tensor]
    # paddle: bias_v: Optional[paddle.Tensor]
    bias_v: Optional[torch.Tensor]

    # paddle: def __init__(
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        is_export=False,
    ) -> None:
        # paddle: super(MyMultiheadAttention, self).__init__()
        super(MyMultiheadAttention, self).__init__()
        # paddle: self.embed_dim = embed_dim
        self.embed_dim = embed_dim
        # paddle: self.kdim = kdim if kdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        # paddle: self.vdim = vdim if vdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        # paddle: self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        # paddle: self.num_heads = num_heads
        self.num_heads = num_heads
        # paddle: self.dropout = dropout
        self.dropout = dropout
        # paddle: self.batch_first = batch_first
        self.batch_first = batch_first
        # paddle: self.head_dim = embed_dim // num_heads
        self.head_dim = embed_dim // num_heads
        # paddle: self.is_export = is_export
        self.is_export = is_export
        # paddle: assert (
        # paddle:     self.head_dim * num_heads == self.embed_dim
        # paddle: ), "embed_dim must be divisible by num_heads"
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # paddle: if self._qkv_same_embed_dim is False:
        if not self._qkv_same_embed_dim:
            # paddle: pass
            # This path is not implemented in the paddle code, so we replicate that.
             raise NotImplementedError("Separate projection weights are not implemented.")
        # paddle: else:
        else:
            # paddle: if dtype is None:
            # paddle:     dtype = paddle.float32
            # paddle: self.in_proj_weight = paddle.create_parameter(
            # paddle:     (3 * embed_dim, embed_dim), dtype
            # paddle: )
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim, device=device, dtype=dtype))
            # paddle: self.q_proj_weight = None
            self.q_proj_weight = None
            # paddle: self.k_proj_weight = None
            self.k_proj_weight = None
            # paddle: self.v_proj_weight = None
            self.v_proj_weight = None

        # paddle: if bias:
        if bias:
            # paddle: self.in_proj_bias = paddle.create_parameter((3 * embed_dim,), dtype)
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, device=device, dtype=dtype))
            # paddle: zeros_(self.in_proj_bias)
        # paddle: else:
        else:
            # paddle: self.in_proj_bias = None
            self.register_parameter('in_proj_bias', None)
        # paddle: self.out_proj = nn.Linear(embed_dim, embed_dim, bias_attr=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # paddle: if add_bias_kv:
        if add_bias_kv:
            # paddle: pass
            pass
        # paddle: else:
        else:
            # paddle: self.bias_k = self.bias_v = None
            self.bias_k = self.bias_v = None

        # paddle: self.add_zero_attn = add_zero_attn
        self.add_zero_attn = add_zero_attn

        # paddle: self._reset_parameters()
        self._reset_parameters()

    # paddle: def _reset_parameters(self):
    def _reset_parameters(self):
        # paddle: if self._qkv_same_embed_dim:
        if self._qkv_same_embed_dim:
            # paddle: xavier_uniform_(self.in_proj_weight)
            xavier_uniform_(self.in_proj_weight)
        # paddle: else:
        else:
            # paddle: xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.q_proj_weight)
            # paddle: xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            # paddle: xavier_uniform_(self.v_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        # paddle: if self.in_proj_bias is not None:
        if self.in_proj_bias is not None:
            # paddle: zeros_(self.in_proj_bias)
            zeros_(self.in_proj_bias)
            # paddle: zeros_(self.out_proj.bias)
            if self.out_proj.bias is not None:
                zeros_(self.out_proj.bias)
        # paddle: if self.bias_k is not None:
        if self.bias_k is not None:
            # paddle: xavier_normal_(self.bias_k)
            xavier_normal_(self.bias_k)
        # paddle: if self.bias_v is not None:
        if self.bias_v is not None:
            # paddle: xavier_normal_(self.bias_v)
            xavier_normal_(self.bias_v)

    # paddle: def forward(
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # The logic seems to imply batch_first=False from the multi_head_attention_forward function
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # paddle: attn_output, attn_output_weights = multi_head_attention_forward(
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=not self._qkv_same_embed_dim,
        )
        
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights

        # paddle: return attn_output, attn_output_weights
        return attn_output, attn_output_weights


# paddle: class LogitsProcessorList(list):
class LogitsProcessorList(list):
    # paddle: def __call__(self, input_ids, scores, **kwargs):
    def __call__(self, input_ids, scores, **kwargs):
        # paddle: for processor in self:
        for processor in self:
            # paddle: function_args = inspect.signature(processor.__call__).parameters
            function_args = inspect.signature(processor.__call__).parameters
            # paddle: if len(function_args) > 2:
            if len(function_args) > 2:
                # paddle: if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    # paddle: raise ValueError(
                    # paddle:     f"Make sure that all the required parameters: {list(function_args.keys())} for "
                    # paddle:     f"{processor.__class__} are passed to the logits processor."
                    # paddle: )
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                # paddle: scores = processor(input_ids, scores, **kwargs)
                scores = processor(input_ids, scores, **kwargs)
            # paddle: else:
            else:
                # paddle: scores = processor(input_ids, scores)
                scores = processor(input_ids, scores)
        # paddle: return scores
        return scores


# paddle: class ForcedEOSTokenLogitsProcessor(object):
class ForcedEOSTokenLogitsProcessor(object):
    # paddle: def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
    def __init__(self, max_length: int, eos_token_id: Union[int, List[int]]):
        # paddle: self.max_length = max_length
        self.max_length = max_length
        # paddle: if isinstance(eos_token_id, int):
        if isinstance(eos_token_id, int):
            # paddle: eos_token_id = [eos_token_id]
            eos_token_id = [eos_token_id]
        # paddle: self.eos_token_id = eos_token_id
        self.eos_token_id = eos_token_id

    # paddle: def __call__(self, input_ids, scores):
    def __call__(self, input_ids, scores):
        # paddle: cur_len = input_ids.shape[-1]
        cur_len = input_ids.size(-1)
        # paddle: scores_processed = scores
        scores_processed = scores
        # paddle: if cur_len == self.max_length - 1:
        if cur_len == self.max_length - 1:
            # paddle: scores_processed = paddle.full_like(scores, -math.inf)
            scores_processed = torch.full_like(scores, -math.inf)
            # paddle: scores_processed[:, self.eos_token_id] = 0
            scores_processed[:, self.eos_token_id] = 0
        # paddle: return scores_processed
        return scores_processed


# paddle: @dataclass
# paddle: class CausalLMOutputWithCrossAttentions(ModelOutput):
@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):
    # paddle: loss = None
    loss: Optional[torch.FloatTensor] = None
    # paddle: logits = None
    logits: torch.FloatTensor = None
    # paddle: past_key_values = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    # paddle: hidden_states = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # paddle: attentions = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # paddle: cross_attentions = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # paddle: def __init__(self, *args, **kwargs):
    def __init__(self, *args, **kwargs):
        # paddle: super().__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)


# paddle: @dataclass
# paddle: class CausalLMOutputWithCrossAttentionsAndCounting(ModelOutput):
@dataclass
class CausalLMOutputWithCrossAttentionsAndCounting(ModelOutput):
    # paddle: logits = None
    logits: Optional[torch.FloatTensor] = None
    # paddle: counting = None
    counting: Optional[torch.FloatTensor] = None
    # paddle: past_key_values = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    # paddle: hidden_states = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # paddle: attentions = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # paddle: cross_attentions = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

    # paddle: def __init__(self, *args, **kwargs):
    def __init__(self, *args, **kwargs):
        # paddle: super().__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)


# paddle: class CustomMBartDecoder(MBartDecoder):
class CustomMBartDecoder(MBartDecoder):
    # paddle: def __init__(self, config):
    def __init__(self, config):
        # paddle: super().__init__(config)
        super().__init__(config)
        # paddle: hidden_size = config.d_model
        hidden_size = config.d_model
        # paddle: self.is_export = config.is_export
        self.is_export = config.is_export
        # paddle: self.counting_context_weight = nn.Sequential(
        self.counting_context_weight = nn.Sequential(
            # paddle: nn.Linear(config.vocab_size, hidden_size),
            nn.Linear(config.vocab_size, hidden_size),
            # paddle: nn.ReLU(),
            nn.ReLU(),
            # paddle: nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            # paddle: nn.ReLU(),
            nn.ReLU(),
            # paddle: nn.Linear(hidden_size, config.d_model),
            nn.Linear(hidden_size, config.d_model),
        )

    # paddle: def forward(
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        count_pred=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # paddle: self.is_export = False if self.training else True
        self.is_export = not self.training
        # paddle: output_attentions = (
        # paddle:     output_attentions
        # paddle:     if output_attentions is not None
        # paddle:     else self.config.output_attentions
        # paddle: )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # paddle: output_hidden_states = (
        # paddle:     output_hidden_states
        # paddle:     if output_hidden_states is not None
        # paddle:     else self.config.output_hidden_states
        # paddle: )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # paddle: use_cache = use_cache if use_cache is not None else self.config.use_cache
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # paddle: return_dict = (
        # paddle:     return_dict if return_dict is not None else self.config.use_return_dict
        # paddle: )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # paddle: if input_ids is not None and inputs_embeds is not None:
        if input_ids is not None and inputs_embeds is not None:
            # paddle: raise ValueError(
            # paddle:     "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            # paddle: )
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        # paddle: elif input_ids is not None:
        elif input_ids is not None:
            # paddle: input = input_ids
            input = input_ids
            # paddle: input_shape = input.shape
            input_shape = input.size()
            # paddle: input_ids = input_ids.reshape([-1, input_shape[-1]])
            input_ids = input_ids.view(-1, input_shape[-1])
        # paddle: elif inputs_embeds is not None:
        elif inputs_embeds is not None:
            # paddle: input_shape = inputs_embeds.shape[:-1]
            input_shape = inputs_embeds.size()[:-1]
            # paddle: input = inputs_embeds[:, :, -1]
            input = inputs_embeds[:, :, -1]
        # paddle: else:
        else:
            # paddle: raise ValueError(
            # paddle:     "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            # paddle: )
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # paddle: past_key_values_length = (
        # paddle:     past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # paddle: )
        past_key_values_length = (
            past_key_values[0][0].size(2) if past_key_values is not None else 0
        )

        # paddle: if inputs_embeds is None:
        if inputs_embeds is None:
            # paddle: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        # paddle: if self._use_flash_attention_2:
        if self._use_flash_attention_2:
            # paddle: attention_mask = (
            # paddle:     attention_mask
            # paddle:     if (attention_mask is not None and 0 in attention_mask)
            # paddle:     else None
            # paddle: )
            attention_mask = (
                attention_mask
                if (attention_mask is not None and (attention_mask == 0).any())
                else None
            )
        # paddle: else:
        else:
            # paddle: if self.is_export:
            if self.is_export:
                # paddle: attention_mask = _prepare_4d_causal_attention_mask_export(
                # paddle:     attention_mask,
                # paddle:     input_shape,
                # paddle:     inputs_embeds,
                # paddle:     past_key_values_length,
                # paddle:     is_export=self.is_export,
                # paddle: ).cast(paddle.float32)
                attention_mask = _prepare_4d_causal_attention_mask_export(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    is_export=self.is_export,
                ).to(torch.float32)
            # paddle: else:
            else:
                # paddle: attention_mask = _prepare_4d_causal_attention_mask(
                # paddle:     attention_mask,
                # paddle:     input_shape,
                # paddle:     inputs_embeds,
                # paddle:     past_key_values_length,
                # paddle:     is_export=self.is_export,
                # paddle: )
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    is_export=self.is_export,
                )

        # paddle: if encoder_hidden_states is not None and encoder_attention_mask is not None:
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # paddle: if self._use_flash_attention_2:
            if self._use_flash_attention_2:
                # paddle: encoder_attention_mask = (
                # paddle:     encoder_attention_mask if 0 in encoder_attention_mask else None
                # paddle: )
                encoder_attention_mask = (
                    encoder_attention_mask if (encoder_attention_mask == 0).any() else None
                )
            # paddle: else:
            else:
                # paddle: encoder_attention_mask = _prepare_4d_attention_mask(
                # paddle:     encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                # paddle: )
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        # paddle: positions = self.embed_positions(input, past_key_values_length)
        # `input` here refers to input_ids in the paddle version
        positions = self.embed_positions(input_ids.view(*input_shape), past_key_values_length)

        # paddle: hidden_states = inputs_embeds + positions
        hidden_states = inputs_embeds + positions

        # paddle: if count_pred is not None:
        if count_pred is not None:
            # paddle: count_context_weight = self.counting_context_weight(count_pred)
            count_context_weight = self.counting_context_weight(count_pred)
            # paddle: hidden_states = hidden_states + 0.5 * count_context_weight.unsqueeze(1)
            hidden_states = hidden_states + 0.5 * count_context_weight.unsqueeze(1)

        # paddle: hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = self.layernorm_embedding(hidden_states)
        # paddle: hidden_states = nn.functional.dropout(
        # paddle:     hidden_states, p=self.dropout, training=self.training
        # paddle: )
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training
        )

        # paddle: if self.gradient_checkpointing and self.training:
        if self.gradient_checkpointing and self.training:
            # paddle: if use_cache:
            if use_cache:
                # paddle: print(
                # paddle:     "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                # paddle: )
                warnings.warn(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                # paddle: use_cache = False
                use_cache = False

        # paddle: all_hidden_states = () if output_hidden_states else None
        all_hidden_states = () if output_hidden_states else None
        # paddle: all_self_attns = () if output_attentions else None
        all_self_attns = () if output_attentions else None
        # paddle: all_cross_attentions = (
        # paddle:     () if (output_attentions and encoder_hidden_states is not None) else None
        # paddle: )
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        # paddle: next_decoder_cache = () if use_cache else None
        next_decoder_cache = () if use_cache else None

        # paddle: for attn_mask, mask_name in zip(
        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            # paddle: if attn_mask is not None:
            if attn_mask is not None:
                # paddle: if attn_mask.size()[0] != len(self.layers):
                if attn_mask.size()[0] != len(self.layers):
                    # paddle: raise ValueError(
                    # paddle:     f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                    # paddle:     f" {attn_mask.size()[0]}."
                    # paddle: )
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                    )

        # paddle: for idx, decoder_layer in enumerate(self.layers):
        for idx, decoder_layer in enumerate(self.layers):
            # paddle: if output_hidden_states:
            if output_hidden_states:
                # paddle: all_hidden_states += (hidden_states,)
                all_hidden_states += (hidden_states,)
            # paddle: if self.training:
            if self.training:
                # paddle: dropout_probability = paddle.rand([])
                dropout_probability = torch.rand(1)
                # paddle: if dropout_probability < self.layerdrop:
                if dropout_probability < self.layerdrop:
                    # paddle: continue
                    continue

            # paddle: past_key_value = (
            # paddle:     past_key_values[idx] if past_key_values is not None else None
            # paddle: )
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            # paddle: if self.gradient_checkpointing and self.training:
            if self.gradient_checkpointing and self.training:
                # paddle: layer_outputs = self._gradient_checkpointing_func(
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions=output_attentions, use_cache=use_cache)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                )
            # paddle: else:
            else:
                # paddle: layer_outputs = decoder_layer(
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx]
                        if cross_attn_head_mask is not None
                        else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            # paddle: hidden_states = layer_outputs[0]
            hidden_states = layer_outputs[0]
            # paddle: if self.is_export:
            if self.is_export:
                # paddle: next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            # paddle: else:
            else:
                # paddle: if use_cache:
                if use_cache:
                    # paddle: next_decoder_cache += (
                    # paddle:     layer_outputs[3 if output_attentions else 1],
                    # paddle: )
                    next_decoder_cache += (
                        layer_outputs[3 if output_attentions else 1],
                    )

            # paddle: if output_attentions:
            if output_attentions:
                # paddle: all_self_attns += (layer_outputs[1],)
                all_self_attns += (layer_outputs[1],)

                # paddle: if encoder_hidden_states is not None:
                if encoder_hidden_states is not None:
                    # paddle: all_cross_attentions += (layer_outputs[2],)
                    all_cross_attentions += (layer_outputs[2],)

        # paddle: hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.layer_norm(hidden_states)

        # paddle: if output_hidden_states:
        if output_hidden_states:
            # paddle: all_hidden_states += (hidden_states,)
            all_hidden_states += (hidden_states,)
        # paddle: if self.is_export:
        if self.is_export:
            # paddle: next_cache = next_decoder_cache
            next_cache = next_decoder_cache
        # paddle: else:
        else:
            # paddle: next_cache = next_decoder_cache if use_cache else None
            next_cache = next_decoder_cache if use_cache else None
        # paddle: if not self.is_export:
        if not self.is_export:
            # paddle: if not return_dict:
            if not return_dict:
                # paddle: return tuple(
                return tuple(
                    v
                    for v in [
                        hidden_states,
                        next_cache,
                        all_hidden_states,
                        all_self_attns,
                        all_cross_attentions,
                    ]
                    if v is not None
                )
        # paddle: return BaseModelOutputWithPastAndCrossAttentions(
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


# paddle: class SelfAttentionBlock(nn.Layer):
class SelfAttentionBlock(nn.Module):
    # paddle: def __init__(self, embed_size, num_heads, is_export):
    def __init__(self, embed_size, num_heads, is_export):
        # paddle: super(SelfAttentionBlock, self).__init__()
        super(SelfAttentionBlock, self).__init__()
        # paddle: self.self_attention = MyMultiheadAttention(
        # paddle:     embed_dim=embed_size, num_heads=num_heads, is_export=is_export
        # paddle: )
        # batch_first=True is more common in PyTorch and simplifies things. 
        # Here we set it to False to match the expected input format (seq, batch, feature)
        # of the custom multi_head_attention_forward function.
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=num_heads, batch_first=False,
        )
        # paddle: self.norm = nn.LayerNorm(embed_size)
        self.norm = nn.LayerNorm(embed_size)

    # paddle: def forward(self, x):
    def forward(self, x):
        # paddle: attn_output, _ = self.self_attention(x, x, x)
        # PyTorch MHA expects (seq, batch, feature) by default
        attn_output, _ = self.self_attention(x, x, x)
        # paddle: x = self.norm(attn_output + x)
        x = self.norm(attn_output + x)
        # paddle: return x
        return x


# paddle: class SeqCountingDecoder(nn.Layer):
class SeqCountingDecoder(nn.Module):
    # paddle: def __init__(
    def __init__(
        self, in_features, out_features, num_heads=8, num_layers=4, is_export=False
    ):
        # paddle: super(SeqCountingDecoder, self).__init__()
        super(SeqCountingDecoder, self).__init__()

        # paddle: self.attention_blocks = nn.LayerList(
        self.attention_blocks = nn.ModuleList(
            [
                # paddle: SelfAttentionBlock(
                # paddle:     embed_size=in_features, num_heads=num_heads, is_export=is_export
                # paddle: )
                SelfAttentionBlock(
                    embed_size=in_features, num_heads=num_heads, is_export=is_export
                )
                for i in range(num_layers)
            ]
        )
        # paddle: self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc1 = nn.Linear(in_features, in_features // 2)
        # paddle: self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        # paddle: self.global_avg_pool = nn.AdaptiveAvgPool1D(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # paddle: self.fc2 = nn.Linear(in_features // 2, out_features)
        self.fc2 = nn.Linear(in_features // 2, out_features)

    # paddle: def forward(self, x):
    def forward(self, x):
        # Input to attention blocks should be (seq, batch, feature)
        x = x.permute(1, 0, 2)
        
        # paddle: for block in self.attention_blocks:
        for block in self.attention_blocks:
            # paddle: x = block(x)
            x = block(x)
            
        # Revert to (batch, seq, feature) for linear layers
        x = x.permute(1, 0, 2)
        
        # paddle: x = self.fc1(x)
        x = self.fc1(x)
        # paddle: x = self.relu(x)
        x = self.relu(x)
        # paddle: x = x.transpose([0, 2, 1])
        x = x.permute(0, 2, 1)
        # paddle: x = self.global_avg_pool(x)
        x = self.global_avg_pool(x)
        # paddle: x = x.squeeze(-1)
        x = x.squeeze(-1)
        # paddle: x = self.fc2(x)
        x = self.fc2(x)
        # paddle: return x
        return x


# paddle: class CustomMBartForCausalLM(MBartForCausalLM):
class CustomMBartForCausalLM(MBartForCausalLM):
    # paddle: def __init__(self, config, length_aware=True):
    def __init__(self, config, length_aware=True):
        # paddle: super().__init__(config)
        super().__init__(config)
        # paddle: self.model.decoder = CustomMBartDecoder(config)
        self.model.decoder = CustomMBartDecoder(config)
        # paddle: self.counting_decoder = SeqCountingDecoder(
        # paddle:     config.d_model, config.vocab_size, is_export=config.is_export
        # paddle: )
        self.counting_decoder = SeqCountingDecoder(
            config.d_model, config.vocab_size, is_export=config.is_export
        )
        # paddle: self.length_aware = length_aware
        self.length_aware = length_aware

    # paddle: def forward(
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        count_gt=None,
    ):
        # paddle: output_attentions = (
        # paddle:     output_attentions
        # paddle:     if output_attentions is not None
        # paddle:     else self.config.output_attentions
        # paddle: )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        # paddle: output_hidden_states = (
        # paddle:     output_hidden_states
        # paddle:     if output_hidden_states is not None
        # paddle:     else self.config.output_hidden_states
        # paddle: )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        # paddle: return_dict = (
        # paddle:     return_dict if return_dict is not None else self.config.use_return_dict
        # paddle: )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # paddle: if self.length_aware:
        if self.length_aware:
            # paddle: count_pred = self.counting_decoder(encoder_hidden_states)
            count_pred = self.counting_decoder(encoder_hidden_states)
        # paddle: else:
        else:
            # paddle: count_pred = None
            count_pred = None

        # paddle: outputs = self.model.decoder(
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            count_pred=count_pred,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # paddle: logits = self.lm_head(outputs[0])
        last_hidden_state = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.lm_head(last_hidden_state)

        # paddle: return CausalLMOutputWithCrossAttentionsAndCounting(
        return CausalLMOutputWithCrossAttentionsAndCounting(
            logits=logits,
            counting=count_pred,
            past_key_values=outputs.past_key_values if return_dict else outputs[1],
            hidden_states=outputs.hidden_states if return_dict else outputs[2],
            attentions=outputs.attentions if return_dict else outputs[3],
            cross_attentions=outputs.cross_attentions if return_dict else outputs[4],
        )


# paddle: class UniMERNetHead(nn.Layer):
class UniMERNetHead(nn.Module):
    # paddle: def __init__(
    def __init__(
        self,
        max_new_tokens=1536,
        decoder_start_token_id=0,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        in_channels=1024,
        encoder_hidden_size=1024,
        decoder_hidden_size=1024,
        decoder_ffn_dim=4096,
        decoder_layers=8,
        is_export=False,
        length_aware=True,
    ):
        # paddle: super().__init__()
        super().__init__()
        # paddle: mbart_config_dict = { ... }
        mbart_config_dict = {
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "add_cross_attention": True,
            "add_final_layer_norm": True,
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "classifier_dropout": 0.0,
            "d_model": decoder_hidden_size,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": decoder_ffn_dim,
            "decoder_layerdrop": 0.0,
            "decoder_layers": decoder_layers,
            "dropout": 0.1,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 12,
            "eos_token_id": 2,
            "forced_eos_token_id": 2,
            "init_std": 0.02,
            "is_decoder": True,
            "is_encoder_decoder": False,
            "output_hidden_states": False,
            "max_position_embeddings": max_new_tokens,
            "model_type": "mbart",
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "scale_embedding": True,
            "tie_word_embeddings": False,
            "transformers_version": "4.40.0",
            "use_cache": True,
            "use_return_dict": True,
            "vocab_size": 50000,
            "_attn_implementation": "eager",
            "hidden_size": decoder_hidden_size,
            "is_export": is_export,
        }

        # paddle: self.max_new_tokens = max_new_tokens
        self.max_new_tokens = max_new_tokens
        # paddle: self.decoder_start_token_id = decoder_start_token_id
        self.decoder_start_token_id = decoder_start_token_id
        # paddle: self.temperature = temperature
        self.temperature = temperature
        # paddle: self.do_sample = do_sample
        self.do_sample = do_sample
        # paddle: self.top_p = top_p
        self.top_p = top_p
        # paddle: self.max_seq_len = max_new_tokens
        self.max_seq_len = max_new_tokens
        # paddle: self.config_decoder = MBartConfig(**mbart_config_dict)
        self.config_decoder = MBartConfig(**mbart_config_dict)
        # paddle: self.encoder_hidden_size = encoder_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        # paddle: self.is_export = self.config_decoder.is_export
        self.is_export = self.config_decoder.is_export
        # paddle: self.decoder = CustomMBartForCausalLM(
        # paddle:     self.config_decoder, length_aware=length_aware
        # paddle: )
        self.decoder = CustomMBartForCausalLM(
            self.config_decoder, length_aware=length_aware
        )
        # paddle: if self.config_decoder.hidden_size != self.encoder_hidden_size:
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            # paddle: self.enc_to_dec_proj = nn.Linear(
            # paddle:     self.encoder_hidden_size, self.config_decoder.hidden_size
            # paddle: )
            self.enc_to_dec_proj = nn.Linear(
                self.encoder_hidden_size, self.config_decoder.hidden_size
            )
        # paddle: generation_config = { ... }
        generation_config = {
            "max_length": 1537,
            "forced_eos_token_id": 2,
        }
        # paddle: self.eos_token_id = generation_config["forced_eos_token_id"]
        self.eos_token_id = generation_config["forced_eos_token_id"]
        # paddle: self.pad_token_id = self.config_decoder.pad_token_id
        self.pad_token_id = self.config_decoder.pad_token_id
        # paddle: self.logits_processor = LogitsProcessorList()
        self.logits_processor = LogitsProcessorList()
        # paddle: self.logits_processor.append(
        # paddle:     ForcedEOSTokenLogitsProcessor( ... )
        # paddle: )
        self.logits_processor.append(
            ForcedEOSTokenLogitsProcessor(
                generation_config["max_length"],
                generation_config["forced_eos_token_id"],
            )
        )

    # paddle: def _get_decoder_start_token_id(
    def _get_decoder_start_token_id(
        self, decoder_start_token_id=None, bos_token_id=None
    ) -> int:
        # A simplified version, as `generation_config` is not a class attribute in this conversion
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else self.decoder_start_token_id
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config_decoder.bos_token_id

        # paddle: if decoder_start_token_id is not None:
        if decoder_start_token_id is not None:
            # paddle: return decoder_start_token_id
            return decoder_start_token_id
        # paddle: elif bos_token_id is not None:
        elif bos_token_id is not None:
            # paddle: return bos_token_id
            return bos_token_id
        # paddle: raise ValueError(...)
        raise ValueError(
            "`decoder_start_token_id` or `bos_token_id` has to be defined for encoder-decoder generation."
        )

    # paddle: def _prepare_decoder_input_ids_for_generation(
    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_kwargs,
        device,
        decoder_start_token_id=None,
        bos_token_id=None,
    ):
        # paddle: if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            # paddle: decoder_input_ids = model_kwargs.pop("decoder_input_ids")
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        # paddle: elif "input_ids" in model_kwargs:
        elif "input_ids" in model_kwargs:
            # paddle: decoder_input_ids = model_kwargs.pop("input_ids")
            decoder_input_ids = model_kwargs.pop("input_ids")
        # paddle: else:
        else:
            # paddle: decoder_input_ids = None
            decoder_input_ids = None

        # paddle: decoder_start_token_id = self._get_decoder_start_token_id(...)
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id
        )
        
        # paddle: if isinstance(decoder_start_token_id, list):
        if isinstance(decoder_start_token_id, list):
            # paddle: if len(decoder_start_token_id) != batch_size:
            if len(decoder_start_token_id) != batch_size:
                # paddle: raise ValueError(...)
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            # paddle: decoder_input_ids_start = paddle.to_tensor(...)
            decoder_input_ids_start = torch.tensor(
                decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )
            # paddle: decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        # paddle: else:
        else:
            # paddle: decoder_input_ids_start = (paddle.ones(...) * decoder_start_token_id)
            decoder_input_ids_start = (
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.long,
                    device=device
                ) * decoder_start_token_id
            )
        
        # paddle: if decoder_input_ids is None:
        if decoder_input_ids is None:
            # paddle: decoder_input_ids = decoder_input_ids_start
            decoder_input_ids = decoder_input_ids_start
        # ... (skipping some model-specific elifs)
        # paddle: elif (isinstance(decoder_start_token_id, int) ... )
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all().item()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item()
        ):
            # paddle: decoder_input_ids = paddle.concat(...)
            decoder_input_ids = torch.cat(
                [decoder_input_ids_start, decoder_input_ids], dim=-1
            )
            # paddle: if "decoder_attention_mask" in model_kwargs:
            if "decoder_attention_mask" in model_kwargs:
                # paddle: decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                # paddle: decoder_attention_mask = paddle.cat(...)
                decoder_attention_mask = torch.cat(
                    (
                        torch.ones_like(decoder_attention_mask)[:, :1],
                        decoder_attention_mask,
                    ),
                    dim=-1,
                )
                # paddle: model_kwargs["decoder_attention_mask"] = decoder_attention_mask
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        # paddle: return decoder_input_ids, model_kwargs
        return decoder_input_ids, model_kwargs

    # paddle: def prepare_inputs_for_generation_mbart(
    def prepare_inputs_for_generation_mbart(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        # paddle: if attention_mask is None:
        if attention_mask is None:
            # paddle: attention_mask = paddle.ones(input_ids.shape)
            attention_mask = torch.ones_like(input_ids)

        # paddle: if past_key_values:
        if past_key_values:
            # paddle: past_length = past_key_values[0][0].shape[2]
            past_length = past_key_values[0][0].size(2)

            # paddle: if input_ids.shape[1] > past_length:
            if input_ids.size(1) > past_length:
                # paddle: remove_prefix_length = past_length
                remove_prefix_length = past_length
            # paddle: else:
            else:
                # paddle: remove_prefix_length = input_ids.shape[1] - 1
                remove_prefix_length = input_ids.size(1) - 1

            # paddle: input_ids = input_ids[:, remove_prefix_length:]
            input_ids = input_ids[:, remove_prefix_length:]
        # paddle: return { ... }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    # paddle: def prepare_inputs_for_generation(
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # paddle: decoder_inputs = self.prepare_inputs_for_generation_mbart(...)
        decoder_inputs = self.prepare_inputs_for_generation_mbart(
            input_ids, past_key_values=past_key_values
        )
        # paddle: decoder_attention_mask = ...
        decoder_attention_mask = (
            decoder_inputs["attention_mask"]
            if "attention_mask" in decoder_inputs
            else None
        )
        # paddle: input_dict = { ... }
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        # paddle: return input_dict
        return input_dict

    # paddle: def prepare_inputs_for_generation_export(
    def prepare_inputs_for_generation_export(
        self,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # paddle: input_dict = { ... }
        input_dict = {
            "decoder_attention_mask": None,
            "use_cache": use_cache,
        }
        # paddle: return input_dict
        return input_dict

    # paddle: def _extract_past_from_model_output(
    def _extract_past_from_model_output(
        self, outputs: ModelOutput, standardize_cache_format: bool = False
    ):
        # paddle: past_key_values = None
        past_key_values = None
        # paddle: if "past_key_values" in outputs:
        if "past_key_values" in outputs:
            # paddle: past_key_values = outputs.past_key_values
            past_key_values = outputs.past_key_values
        # paddle: elif "mems" in outputs:
        elif "mems" in outputs:
            # paddle: past_key_values = outputs.mems
            past_key_values = outputs.mems
        # paddle: elif "past_buckets_states" in outputs:
        elif "past_buckets_states" in outputs:
            # paddle: past_key_values = outputs.past_buckets_states
            past_key_values = outputs.past_buckets_states

        # paddle: return past_key_values
        return past_key_values

    # paddle: def _update_model_kwargs_for_generation(
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # paddle: model_kwargs["past_key_values"] = self._extract_past_from_model_output(...)
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        # paddle: if getattr(outputs, "state", None) is not None:
        if getattr(outputs, "state", None) is not None:
            # paddle: model_kwargs["state"] = outputs.state
            model_kwargs["state"] = outputs.state

        # paddle: if "token_type_ids" in model_kwargs:
        if "token_type_ids" in model_kwargs:
            # paddle: token_type_ids = model_kwargs["token_type_ids"]
            token_type_ids = model_kwargs["token_type_ids"]
            # paddle: model_kwargs["token_type_ids"] = paddle.concat(...)
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )
        
        # paddle: if not is_encoder_decoder:
        if not is_encoder_decoder:
            # paddle: if "attention_mask" in model_kwargs:
            if "attention_mask" in model_kwargs:
                # paddle: attention_mask = model_kwargs["attention_mask"]
                attention_mask = model_kwargs["attention_mask"]
                # paddle: model_kwargs["attention_mask"] = paddle.concat(...)
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype),
                    ],
                    dim=-1,
                )
        # paddle: else:
        else:
            # paddle: if "decoder_attention_mask" in model_kwargs:
            if "decoder_attention_mask" in model_kwargs:
                # paddle: decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                # paddle: model_kwargs["decoder_attention_mask"] = paddle.concat(...)
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        torch.ones((decoder_attention_mask.size(0), 1), device=decoder_attention_mask.device, dtype=decoder_attention_mask.dtype),
                    ],
                    dim=-1,
                )

        # paddle: if ( "cache_position" in model_kwargs ... )
        if (
            "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            # paddle: model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
        
        # paddle: return model_kwargs
        return model_kwargs

    # paddle: def stopping_criteria(self, input_ids):
    def stopping_criteria(self, input_ids):
        # paddle: if self.is_export:
        if self.is_export:
            # paddle: return input_ids[:, -1] == paddle.to_tensor([self.eos_token_id])
            return input_ids[:, -1] == torch.tensor([self.eos_token_id], device=input_ids.device)
        # paddle: is_done = paddle.isin(input_ids[:, -1], paddle.to_tensor([self.eos_token_id]))
        # `torch.isin` is available in recent PyTorch versions.
        is_done = torch.isin(input_ids[:, -1], torch.tensor([self.eos_token_id], device=input_ids.device))
        # paddle: return is_done
        return is_done

    # paddle: def generate_single_iter(
    def generate_single_iter(
        self,
        input_ids=None, # Renamed from decoder_input_ids to match decoder's signature
        attention_mask=None, # Renamed from decoder_attention_mask
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None, # Renamed from decoder_inputs_embeds
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # paddle: encoder_hidden_states = encoder_outputs[0]
        # In PyTorch, model outputs are often dataclasses or tuples. Assuming it's a dataclass-like object.
        encoder_hidden_states = encoder_outputs.last_hidden_state
        # paddle: if self.config_decoder.hidden_size != self.encoder_hidden_size:
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            # paddle: encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        # paddle: kwargs_decoder = {}
        kwargs_decoder = {}

        # paddle: decoder_outputs = self.decoder(...)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            inputs_embeds=inputs_embeds,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # paddle: return Seq2SeqLMOutput(...)
        return Seq2SeqLMOutput(
            loss=None,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    # paddle: @paddle.no_grad()
    # paddle: def generate(
    @torch.no_grad()
    def generate(
        self,
        model_kwargs,
    ):
        # paddle: batch_size = model_kwargs["encoder_outputs"]["last_hidden_state"].shape[0]
        batch_size = model_kwargs["encoder_outputs"].last_hidden_state.size(0)
        device = model_kwargs["encoder_outputs"].last_hidden_state.device
        # paddle: generation_config = { ... }
        generation_config = {
            "decoder_start_token_id": 0,
            "bos_token_id": 0,
        }
        # paddle: input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(...)
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            device=device,
            decoder_start_token_id=generation_config["decoder_start_token_id"],
            bos_token_id=generation_config["bos_token_id"],
        )
        # paddle: model_kwargs["key use_cache"] = True
        model_kwargs["use_cache"] = True # Correcting the key
        # paddle: batch_size, cur_len = input_ids.shape
        batch_size, cur_len = input_ids.size()

        # paddle: if "inputs_embeds" in model_kwargs:
        if "inputs_embeds" in model_kwargs:
            # paddle: cur_len = model_kwargs["inputs_embeds"].shape[1]
            cur_len = model_kwargs["inputs_embeds"].size(1)
        # paddle: model_kwargs["cache_position"] = paddle.arange(cur_len)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=device)
        # paddle: pad_token_id = self.pad_token_id
        pad_token_id = self.pad_token_id
        # paddle: eos_token_id = [self.eos_token_id]
        eos_token_id = [self.eos_token_id]
        # paddle: eos_token = self.eos_token_id
        eos_token = self.eos_token_id
        # paddle: unfinished_sequences = paddle.ones(batch_size, dtype=paddle.int64)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        # paddle: for idx in range(self.max_seq_len):
        for idx in range(self.max_seq_len):
            # paddle: model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            # paddle: outputs = self.generate_single_iter(...)
            outputs = self.generate_single_iter(
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            # paddle: next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = outputs.logits[:, -1, :]

            # paddle: next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            # paddle: next_tokens = paddle.argmax(next_tokens_scores, axis=-1)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            # paddle: if eos_token_id is not None:
            if eos_token_id is not None:
                # paddle: if pad_token_id is None:
                if pad_token_id is None:
                    # paddle: raise ValueError(...)
                    raise ValueError(
                        "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
                    )
                # paddle: next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )
            # paddle: input_ids = paddle.concat([input_ids, next_tokens[:, None]], axis=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            # paddle: model_kwargs = self._update_model_kwargs_for_generation(...)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config_decoder.is_encoder_decoder,
            )
            # paddle: unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(input_ids).cast(paddle.int64)
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(
                input_ids
            ).long()

            # paddle: if (eos_token is not None and (...).all()):
            if (
                eos_token is not None
                and (
                    torch.cumsum((input_ids == eos_token).long(), 1)[:, -1]
                    >= 1
                ).all()
            ):
                # paddle: break
                break

        # paddle: return input_ids
        return input_ids

    # paddle: @paddle.no_grad()
    # paddle: def generate_export(
    @torch.no_grad()
    def generate_export(
        self,
        encoder_outputs,
        model_kwargs,
    ):
        # paddle: batch_size = encoder_outputs["last_hidden_state"].shape[0]
        batch_size = encoder_outputs.last_hidden_state.size(0)
        device = encoder_outputs.last_hidden_state.device
        # paddle: generation_config = { ... }
        generation_config = {
            "decoder_start_token_id": 0,
            "bos_token_id": 0,
        }
        # paddle: input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(...)
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            device=device,
            decoder_start_token_id=generation_config["decoder_start_token_id"],
            bos_token_id=generation_config["bos_token_id"],
        )
        # paddle: input_ids = input_ids.reshape([-1, 1])
        input_ids = input_ids.view(-1, 1)
        # paddle: decoder_input_ids = input_ids
        decoder_input_ids = input_ids
        # paddle: model_kwargs["key use_cache"] = True
        model_kwargs["use_cache"] = True
        # paddle: batch_size, cur_len = input_ids.shape
        batch_size, cur_len = input_ids.size()

        # paddle: if "inputs_embeds" in model_kwargs:
        if "inputs_embeds" in model_kwargs:
            # paddle: cur_len = model_kwargs["inputs_embeds"].shape[1]
            cur_len = model_kwargs["inputs_embeds"].size(1)
        # paddle: cache_position = paddle.arange(cur_len)
        cache_position = torch.arange(cur_len, device=device)
        # paddle: pad_token_id = self.pad_token_id
        pad_token_id = self.pad_token_id
        # paddle: eos_token_id = [self.eos_token_id]
        eos_token_id = [self.eos_token_id]
        # paddle: eos_token = self.eos_token_id
        eos_token = self.eos_token_id
        # paddle: unfinished_sequences = paddle.ones([batch_size], dtype=paddle.int64)
        unfinished_sequences = torch.ones([batch_size], dtype=torch.long, device=device)
        # paddle: i_idx = paddle.full([], 0)
        i_idx = 0
        
        # paddle: past_key_values = []
        # paddle: for i in range(8):
        # paddle:     init_arr = paddle.zeros([batch_size, 16, 0, 64])
        # paddle:     paddle.jit.api.set_dynamic_shape(init_arr, [-1, -1, -1, -1])
        # paddle:     cache = (init_arr, init_arr, init_arr, init_arr)
        # paddle:     past_key_values.append(cache)
        # In PyTorch, this explicit initialization for dynamic shapes is not needed.
        # `past_key_values` will be initialized as None and created on the first iteration.
        past_key_values = None

        # paddle: while i_idx < paddle.to_tensor(self.max_seq_len):
        while i_idx < self.max_seq_len:
            # paddle: model_inputs = self.prepare_inputs_for_generation_export(...)
            # This logic is complex and relies on mbart-specific internal structures.
            # We'll use the standard prepare_inputs_for_generation.
            model_inputs = self.prepare_inputs_for_generation(
                decoder_input_ids, past_key_values=past_key_values, encoder_outputs=encoder_outputs, **model_kwargs
            )
            # paddle: decoder_attention_mask = model_inputs["decoder_attention_mask"]
            decoder_attention_mask = model_inputs["attention_mask"]
            # paddle: decoder_attention_mask = paddle.ones(input_ids.shape)
            # The logic in Paddle seems to re-create the attention mask. We'll use the one from prepare_inputs.
            
            # paddle: paddle.jit.api.set_dynamic_shape(decoder_input_ids, [-1, -1])
            # Dynamic shape setting is implicit in PyTorch tracing/scripting
            
            # paddle: outputs = self.generate_single_iter(...)
            outputs = self.generate_single_iter(
                input_ids=model_inputs['input_ids'],
                attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=model_inputs['past_key_values'],
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            # paddle: next_token_logits = outputs.logits[:, -1, :]
            next_token_logits = outputs.logits[:, -1, :]
            # paddle: next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            # paddle: next_tokens = paddle.argmax(next_tokens_scores, axis=-1)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            
            # paddle: if eos_token_id is not None:
            if eos_token_id is not None:
                # paddle: next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # paddle: input_ids = paddle.concat([input_ids, next_tokens.unsqueeze(1)], axis=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=-1)
            
            # paddle: past_length = past_key_values[0][0].shape[2]
            # paddle: decoder_input_ids = next_tokens.unsqueeze(1)
            decoder_input_ids = next_tokens.unsqueeze(1)
            
            # paddle: past_key_values = outputs.past_key_values
            past_key_values = outputs.past_key_values
            
            # paddle: cache_position = cache_position[-1:] + 1
            cache_position = cache_position[-1:] + 1
            
            # paddle: unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(input_ids).cast(paddle.int64)
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(input_ids).long()
            
            # paddle: if (eos_token is not None and (...).all()):
            if (
                eos_token is not None
                and (torch.cumsum((input_ids == eos_token).long(), 1)[:, -1] >= 1).all()
            ):
                # paddle: break
                break

            # paddle: i_idx += 1
            i_idx += 1
        # paddle: return input_ids
        return input_ids

    # paddle: def forwad_train(
    def forward_train( # Corrected typo from 'forwad'
        self,
        encoder_outputs,
        decoder_input_ids,
        decoder_attention_mask,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        # paddle: labels = decoder_input_ids * 1
        labels = decoder_input_ids.clone()
        # paddle: labels = labels.masked_fill_(labels == self.pad_token_id, -100)
        labels[labels == self.pad_token_id] = -100
        # paddle: input_decoder_input_ids = decoder_input_ids[:, :-1]
        input_decoder_input_ids = decoder_input_ids[:, :-1]
        # paddle: input_decoder_attention_mask = decoder_attention_mask[:, :-1]
        input_decoder_attention_mask = decoder_attention_mask[:, :-1]
        # paddle: encoder_hidden_states = encoder_outputs[0]
        encoder_hidden_states = encoder_outputs[0] # Assuming encoder_outputs is a tuple
        # paddle: if self.config_decoder.hidden_size != self.encoder_hidden_size:
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            # paddle: encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        # paddle: kwargs_decoder = {}
        kwargs_decoder = {}
        # paddle: decoder_outputs = self.decoder(...)
        decoder_outputs = self.decoder(
            input_ids=input_decoder_input_ids,
            attention_mask=input_decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=None,
            inputs_embeds=None,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # paddle: logits = decoder_outputs.logits
        logits = decoder_outputs.logits
        # paddle: count_pred = decoder_outputs.counting
        count_pred = decoder_outputs.counting
        # paddle: return logits, count_pred, labels
        return logits, count_pred, labels

    # paddle: def forward(self, inputs, targets=None):
    def forward(self, inputs, targets=None):
        # paddle: self.is_export = False if self.training else True
        self.is_export = not self.training
        # paddle: if not self.training:
        if not self.training:
            # paddle: encoder_outputs = inputs
            encoder_outputs = inputs
            # paddle: if self.is_export:
            if self.is_export:
                # paddle: model_kwargs = { ... }
                model_kwargs = {
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "use_cache": True,
                }
                # paddle: word_pred = self.generate_export(encoder_outputs, model_kwargs)
                word_pred = self.generate_export(encoder_outputs, model_kwargs)
            # paddle: else:
            else:
                # paddle: model_kwargs = { ... }
                model_kwargs = {
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "use_cache": True,
                    "encoder_outputs": encoder_outputs,
                }
                # paddle: word_pred = self.generate(model_kwargs)
                word_pred = self.generate(model_kwargs)

            # paddle: return word_pred
            return word_pred

        # paddle: encoder_outputs, tgt_seq, mask = inputs
        encoder_outputs, tgt_seq, mask = inputs
        # paddle: logits, count_pred, masked_labels = self.forwad_train(...)
        # The labels for training are the decoder input ids shifted
        labels = tgt_seq[:, 1:].contiguous()
        logits, count_pred, _ = self.forward_train(
            encoder_outputs, tgt_seq, mask
        )
        
        # paddle: return logits, count_pred, masked_labels
        return logits, count_pred, labels
