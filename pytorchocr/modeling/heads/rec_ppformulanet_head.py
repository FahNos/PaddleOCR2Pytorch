import math
import re
import numpy as np
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch import Tensor
from collections import OrderedDict
from typing import Optional, Tuple, Union, List, Dict, Any
from dataclasses import dataclass, fields, is_dataclass

# Giả định các import này tương ứng với phiên bản PyTorch của thư viện
# paddle: from ppocr.modeling.backbones.rec_donut_swin import DonutSwinModelOutput
from pytorchocr.modeling.backbones.rec_donut_swin import DonutSwinModelOutput
# paddle: from ppocr.modeling.heads.rec_unimernet_head import (...)
from pytorchocr.modeling.heads.rec_unimernet_head import (
    MBartForCausalLM,
    MBartDecoder,
    MBartConfig,
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    CausalLMOutputWithCrossAttentions,
    LogitsProcessorList,
    ForcedEOSTokenLogitsProcessor,
    UniMERNetHead,
)


# paddle: @dataclass
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
        past_key_values_length=0,
        sliding_window=None,
        is_export=False,
    ):
        
        bsz, tgt_len = input_ids_shape
        device = input_ids_shape.device if isinstance(input_ids_shape, torch.Tensor) else "cpu"


        if is_export:
            mask = torch.full(
                (tgt_len, tgt_len), torch.finfo(torch.float64).min, dtype=torch.float64, device=device
            )
            mask_cond = torch.arange(mask.shape[-1], device=device)
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0
            )
        else:
            mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
            mask_cond = torch.arange(mask.shape[-1], device=device)
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0
            )
            mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
                dim=-1,
            )

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1
            context_mask = torch.tril(
                torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal
            )
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    @staticmethod
    def _make_causal_mask_parallel(
        input_ids_shape,
        dtype,
        past_key_values_length=0,
        sliding_window=None,
        parallel_step=1,
        is_export=False,
    ):        
        bsz, tgt_len = input_ids_shape
        device = input_ids_shape.device if isinstance(input_ids_shape, torch.Tensor) else "cpu"

        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.shape[-1], device=device)
        mask_cond_parallel = torch.arange(mask.shape[-1], device=device)

        mask_parallel = torch.arange(0, tgt_len, step=parallel_step, device=device).view(1, -1)
        mask_parallel = torch.repeat_interleave(mask_parallel, parallel_step, 1)[:, :tgt_len]
        mask.masked_fill_(
            mask_cond < (mask_parallel + parallel_step).view(mask.shape[-1], 1), 0
        )
        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask],
                dim=-1,
            )

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1
            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)
        
        return mask[None, None, :, :].expand(
            bsz, 1, tgt_len, tgt_len + past_key_values_length
        )

    def to_4d(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        use_parallel=False,
        parallel_step=3,
        is_export=False,
    ):       
        input_shape = (attention_mask_2d.shape[0], query_length)

        causal_4d_mask = None
        if use_parallel:
            step = parallel_step
        else:
            step = 1
        if (input_shape[-1] > step or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length

            # paddle: if use_parallel:
            if use_parallel:
                # paddle: causal_4d_mask = self._make_causal_mask_parallel(...)
                causal_4d_mask = self._make_causal_mask_parallel(
                    input_shape,
                    dtype,
                    past_key_values_length=past_key_values_length,
                    sliding_window=self.sliding_window,
                    parallel_step=parallel_step,
                    is_export=is_export,
                )
            # paddle: else:
            else:
                # paddle: causal_4d_mask = self._make_causal_mask(...)
                causal_4d_mask = self._make_causal_mask(
                    input_shape,
                    dtype,
                    past_key_values_length=past_key_values_length,
                    sliding_window=self.sliding_window,
                    is_export=is_export,
                )

        elif self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is currently only implemented for causal masking"
            )

        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        )

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(
                expanded_attn_mask.to(torch.bool), torch.finfo(dtype).min
            )

        expanded_4d_mask = expanded_attn_mask
        return expanded_4d_mask

    def to_4d_export(
        self,
        attention_mask_2d,
        query_length,
        dtype,
        key_value_length,
        use_parallel=False,
        parallel_step=3,
        is_export=False,
    ):
        input_shape = (attention_mask_2d.shape[0], query_length)

        expanded_attn_mask = self._expand_mask_export(
            attention_mask_2d, dtype, tgt_len=input_shape[-1]
        )
        expanded_4d_mask = expanded_attn_mask

        return expanded_4d_mask

    def _expand_mask(self, mask, dtype, tgt_len=None):
        
        bsz, src_len = mask.shape
  
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )

    def _expand_mask_export(self, mask, dtype, tgt_len=None):
       
        bsz, src_len = mask.shape
        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )

def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
    return AttentionMaskConverter(is_causal=False)._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)

def _prepare_4d_causal_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    use_parallel=False,
    parallel_step=3,
    is_export=False,
):
      
    attn_mask_converter = AttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window
    )
    key_value_length = input_shape[-1] + past_key_values_length

    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            use_parallel=use_parallel,
            parallel_step=parallel_step,
            is_export=is_export,
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(
                inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min
            )
    else:       
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
        )


    return attention_mask

def _prepare_4d_causal_attention_mask_export(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length,
    sliding_window=None,
    use_parallel=False,
    parallel_step=3,
    is_export=False,
):   
    attn_mask_converter = AttentionMaskConverter(
        is_causal=True, sliding_window=sliding_window
    )
    key_value_length = input_shape[-1] + past_key_values_length
    
    shape = attention_mask.shape
    len_shape = len(shape)

    attention_mask = attn_mask_converter.to_4d_export(
        attention_mask,
        input_shape[-1],
        key_value_length=key_value_length,
        dtype=inputs_embeds.dtype,
        use_parallel=use_parallel,
        parallel_step=parallel_step,
        is_export=is_export,
    )
    return attention_mask

class CustomMBartDecoder(MBartDecoder):
    # paddle: def __init__(self, config):
    def __init__(self, config):
        super().__init__(config)
        hidden_size = config.d_model
        self.is_export = config.is_export
        self.config_decoder = config

    # paddle: def forward(...):
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
        self.is_export = not self.training

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input = input_ids
            input_shape = input.shape
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self._use_flash_attention_2:
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        else:
            if self.is_export:
                attention_mask = _prepare_4d_causal_attention_mask_export(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    use_parallel=self.config_decoder.use_parallel,
                    parallel_step=self.config_decoder.parallel_step,
                    is_export=self.is_export,
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    input_shape,
                    inputs_embeds,
                    past_key_values_length,
                    use_parallel=self.config_decoder.use_parallel,
                    parallel_step=self.config_decoder.parallel_step,
                    is_export=self.is_export,
                )

        # paddle: if encoder_hidden_states is not None and encoder_attention_mask is not None:
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = (
                    encoder_attention_mask if 0 in encoder_attention_mask else None
                )
            else:
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )

        positions = self.embed_positions(input, past_key_values_length)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )
        next_decoder_cache = () if use_cache else None

        for attn_mask, mask_name in zip(
            [head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]
        ):
            if attn_mask is not None:
                if attn_mask.size()[0] != len(self.layers):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {attn_mask.size()[0]}."
                        )

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand(())
                if dropout_probability < self.layerdrop:
                    continue
            
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                # Hugging Face PyTorch's _gradient_checkpointing_func khác với Paddle.
                # Cách tiếp cận phổ biến là dùng torch.utils.checkpoint.checkpoint.
                # Tuy nhiên, để giữ logic gần nhất, ta sẽ gọi hàm và giả định nó đã được định nghĩa.
                # layer_outputs = self._gradient_checkpointing_func(...)
                raise NotImplementedError("PyTorch gradient checkpointing needs to be adapted, e.g., using torch.utils.checkpoint.")

            else:
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
            hidden_states = layer_outputs[0]
            
            if self.is_export:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            else:
                if use_cache:
                    next_decoder_cache += (
                        layer_outputs[3 if output_attentions else 1],
                    )
            
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        if self.is_export:
            next_cache = next_decoder_cache
        else:
            next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
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
        
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class CustomMBartForCausalLM(MBartForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model.decoder = CustomMBartDecoder(config)

    # paddle: def forward(...):
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
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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
        logits = self.lm_head(outputs[0])

        return CausalLMOutputWithCrossAttentions(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

class PPFormulaNet_Head(UniMERNetHead):
    def __init__(
        self,
        max_new_tokens=1536,
        decoder_start_token_id=0,
        temperature=0.2,
        do_sample=False,
        top_p=0.95,
        in_channels=1024,
        decoder_layers=8,
        encoder_hidden_size=1024,
        decoder_ffn_dim=4096,
        decoder_hidden_size=1024,
        is_export=False,
        length_aware=True,
        use_parallel=False,
        parallel_step=3,
    ):
        super().__init__()
        
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
            "dropout": 0.1, "encoder_attention_heads": 16, 
            "encoder_ffn_dim": 4096, 
            "encoder_layerdrop": 0.0, 
            "encoder_layers": 12, 
            "eos_token_id": 2, 
            "forced_eos_token_id": 2, 
            "init_std": 0.02, 
            "is_decoder": True, 
            "is_encoder_decoder": False, 
            "output_hidden_states": False, 
            "max_position_embeddings": (
                 max_new_tokens + parallel_step if use_parallel else max_new_tokens
                 ), 
            "model_type": "mbart", 
            "num_hidden_layers": 12, 
            "pad_token_id": 1, 
            "scale_embedding": True, 
            "tie_word_embeddings": False, 
            "transformers_version": "4.40.0", 
            "use_cache": True, 
            "use_return_dict": True, 
            "vocab_size": 50000, 
            "_attn_implementation": 
            "eager", 
            "hidden_size": decoder_hidden_size, 
            "use_parallel": use_parallel, 
            "parallel_step": int(parallel_step), 
            "is_export": is_export,
        }
        self.decoder_start_token_id = decoder_start_token_id
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.is_export = is_export
        self.max_seq_len = max_new_tokens
        self.config_decoder = MBartConfig(**mbart_config_dict)
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder = CustomMBartForCausalLM(self.config_decoder)
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            self.enc_to_dec_proj = nn.Linear(
                self.encoder_hidden_size, self.config_decoder.hidden_size
            )
        generation_config = {"max_length": 1537, "forced_eos_token_id": 2}
        self.eos_token_id = generation_config["forced_eos_token_id"]
        self.pad_token_id = self.config_decoder.pad_token_id
        self.logits_processor = LogitsProcessorList()
        self.logits_processor.append(
            ForcedEOSTokenLogitsProcessor(
                generation_config["max_length"],
                generation_config["forced_eos_token_id"],
            )
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        decoder_inputs = self.prepare_inputs_for_generation_mbart(
            input_ids, past_key_values=past_key_values
        )
        decoder_attention_mask = (
            decoder_inputs["attention_mask"]
            if "attention_mask" in decoder_inputs
            else None
        )
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def _extract_past_from_model_output(
        self, outputs: ModelOutput, standardize_cache_format: bool = False
    ):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states
        return past_key_values

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )
        
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], 1)),
                    ],
                    dim=-1,
                )
        else:
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [
                        decoder_attention_mask,
                        decoder_attention_mask.new_ones(
                            (decoder_attention_mask.shape[0], 1)
                        ),
                    ],
                    dim=-1,
                )
        
        if (
            "cache_position" in model_kwargs
            and model_kwargs["cache_position"] is not None
        ):
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
        return model_kwargs

    def stopping_criteria(self, input_ids):
        if self.is_export:
            return input_ids[:, -1] == torch.tensor([self.eos_token_id], device=input_ids.device)
        is_done = torch.isin(input_ids[:, -1], torch.tensor([self.eos_token_id], device=input_ids.device))
        return is_done

    def stopping_criteria_parallel(self, input_ids):
        parallel_step = self.config_decoder.parallel_step

        if self.is_export:
            is_done_list = []
            for i in range(parallel_step, 0, -1):
                cur_is_done = input_ids[:, -i] == torch.tensor([self.eos_token_id], device=input_ids.device)
                is_done_list.append(cur_is_done)
            is_done_list = torch.stack(is_done_list).transpose(1, 0)
            return is_done_list
        else:
            is_done = torch.isin(
                input_ids[:, -parallel_step:],
                torch.tensor([self.eos_token_id], device=input_ids.device).view(1, 1),
            )
            return is_done 

    def generate_single_iter(
        self,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        encoder_hidden_states = encoder_outputs[0]
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        kwargs_decoder = {}
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
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

    def _prepare_decoder_input_ids_for_generation(
        self,
        batch_size,
        model_kwargs,
        decoder_start_token_id=None,
        bos_token_id=None,
    ):

        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        
        decoder_start_token_id = self._get_decoder_start_token_id(
            decoder_start_token_id, bos_token_id
        )

        device = next(iter(model_kwargs.values())).device if model_kwargs and len(model_kwargs)>0 else "cpu"
        if decoder_input_ids is not None:
            device = decoder_input_ids.device

        if isinstance(decoder_start_token_id, list):
            if len(decoder_start_token_id) != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {len(decoder_start_token_id)}"
                )
            decoder_input_ids_start = torch.tensor(
                decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )
            decoder_input_ids_start = decoder_input_ids_start.view(-1, 1)
        else:
            use_parallel = self.config_decoder.use_parallel
            parallel_step = self.config_decoder.parallel_step

            if use_parallel:
                decoder_input_ids_start = torch.full(
                    (batch_size, parallel_step),
                    decoder_start_token_id,
                    dtype=torch.long,
                    device=device
                )
            else:
                decoder_input_ids_start = torch.full(
                    (batch_size, 1),
                    decoder_start_token_id,
                    dtype=torch.long,
                    device=device
                )

        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        elif (
            hasattr(self, 'config') and hasattr(self.config, 'model_type') and self.config.model_type == "vision-encoder-decoder"
            and hasattr(self, 'name_or_path') and "donut" in self.name_or_path.lower()
        ):
            pass
        elif hasattr(self, 'config') and hasattr(self.config, 'model_type') and self.config.model_type in ["whisper"]:
            pass
        elif (
            isinstance(decoder_start_token_id, int)
            and (decoder_input_ids[:, 0] != decoder_start_token_id).all()
        ) or (
            isinstance(decoder_start_token_id, torch.Tensor)
            and (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all()
        ):
            decoder_input_ids = torch.cat(
                [decoder_input_ids_start, decoder_input_ids], dim=-1
            )
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (
                        torch.ones_like(decoder_attention_mask)[:, :1],
                        decoder_attention_mask,
                    ),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    @torch.no_grad()
    def generate_export(
        self,
        encoder_outputs,
        model_kwargs,
    ):
        use_parallel = self.config_decoder.use_parallel
        parallel_step = self.config_decoder.parallel_step
        batch_size = encoder_outputs["last_hidden_state"].shape[0]
        device = encoder_outputs["last_hidden_state"].device
        generation_config = {"decoder_start_token_id": 0, "bos_token_id": 0}
        
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config["decoder_start_token_id"],
            bos_token_id=generation_config["bos_token_id"],
        )
        if not use_parallel:
            input_ids = input_ids.view(-1, 1)
        decoder_input_ids = input_ids
        model_kwargs["key use_cache"] = True
        batch_size, cur_len = input_ids.shape

        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]

        cache_position = torch.arange(cur_len, device=device)
        pad_token_id = self.pad_token_id
        eos_token_id = [self.eos_token_id]
        eos_token = self.eos_token_id
        
        if use_parallel:
            unfinished_sequences = torch.ones([batch_size, parallel_step], dtype=torch.long, device=device)
            parallel_length = math.ceil(self.max_seq_len / parallel_step)
        else:
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
            parallel_length = self.max_seq_len
        
        i_idx = torch.tensor(0, device=device)
        past_key_values = []
        decoder_attention_heads = self.config_decoder.decoder_attention_heads
        decoder_attention_heads_dim = int(self.config_decoder.d_model / decoder_attention_heads)
        
        for i in range(self.config_decoder.decoder_layers):
            init_arr = torch.zeros(
                batch_size, decoder_attention_heads, 0, decoder_attention_heads_dim, device=device
            )
            cache = (init_arr, init_arr, init_arr, init_arr)
            past_key_values.append(cache)
            
        while i_idx < torch.tensor(parallel_length, device=device):
            model_inputs = self.prepare_inputs_for_generation(
                decoder_input_ids, 
                past_key_values=past_key_values, 
                **model_kwargs
                )
            decoder_input_ids = model_inputs['decoder_input_ids'] # Cập nhật lại từ output
            
            decoder_attention_mask = torch.ones(input_ids.shape, device=device, dtype=torch.long)
            
            
            outputs = self.generate_single_iter(
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                past_key_values=past_key_values,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if use_parallel:
                next_token_logits = outputs.logits[:, -parallel_step:, :]
            else:
                next_token_logits = outputs.logits[:, -1, :]
            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("...")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            if use_parallel:
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
                decoder_input_ids = next_tokens
            else:
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=-1)
                decoder_input_ids = next_tokens.unsqueeze(1)
            
            past_key_values = outputs.past_key_values
            cache_position = cache_position[-1:] + 1
            if use_parallel:
                unfinished_sequences = (
                    unfinished_sequences
                    & ~self.stopping_criteria_parallel(input_ids).to(torch.long)
                )
            else:
                unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(
                    input_ids
                ).to(torch.long)

            if (
                eos_token is not None
                and (
                    torch.cumsum((input_ids == eos_token).long(), 1)[:, -1]
                    >= 1
                ).all()
            ):
                break
            i_idx += 1
        
        return input_ids
    
    @torch.no_grad()
    def generate(
        self,
        encoder_outputs,
        model_kwargs,
    ):
        use_parallel = self.config_decoder.use_parallel
        parallel_step = self.config_decoder.parallel_step
        batch_size = encoder_outputs["last_hidden_state"].shape[0]
        device = encoder_outputs["last_hidden_state"].device
        generation_config = {"decoder_start_token_id": 0, "bos_token_id": 0}
        
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config["decoder_start_token_id"],
            bos_token_id=generation_config["bos_token_id"],
        )

        decoder_input_ids = input_ids 
        model_kwargs["key use_cache"] = True
        batch_size, cur_len = input_ids.shape
        
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        model_kwargs["cache_position"] = torch.arange(cur_len, device=device)
        pad_token_id = self.pad_token_id
        eos_token_id = [self.eos_token_id]
        eos_token = self.eos_token_id

        if use_parallel:
            unfinished_sequences = torch.ones([batch_size, parallel_step], dtype=torch.long, device=device)
            parallel_length = math.ceil(self.max_seq_len / parallel_step)
        else:
            unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
            parallel_length = self.max_seq_len
        past_key_values = [] 

        for idx in range(parallel_length):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self.generate_single_iter(
                **model_inputs,
                encoder_outputs=encoder_outputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            if use_parallel:
                next_token_logits = outputs.logits[:, :, :]
            else:
                next_token_logits = outputs.logits[:, -1, :]
            
            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("...")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            if use_parallel:
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            else:
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config_decoder.is_encoder_decoder,
            )
            if use_parallel:
                unfinished_sequences = (
                    unfinished_sequences
                    & ~self.stopping_criteria_parallel(input_ids).to(torch.long)
                )
            else:
                unfinished_sequences = (
                    unfinished_sequences
                    & ~self.stopping_criteria(input_ids).to(torch.long)
                )

            if (
                eos_token is not None
                and (torch.cumsum((input_ids == eos_token).long(), 1)[:, -1] >= 1).all()
            ):
                break
        return input_ids

    def forwad_train(
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
      
        device = decoder_input_ids.device
        if self.config_decoder.use_parallel:
            batch = decoder_input_ids.shape[0]
            add_sos_token = self.config_decoder.parallel_step - 1
            start_token = torch.zeros(batch, add_sos_token, dtype=torch.long, device=device)
            start_mask = torch.ones(batch, add_sos_token, dtype=torch.long, device=device)
            decoder_input_ids = torch.cat([start_token, decoder_input_ids], dim=1)
            decoder_attention_mask = torch.cat([start_mask, decoder_attention_mask], dim=1)
        
        labels = decoder_input_ids.clone()
        labels.masked_fill_(labels == self.pad_token_id, -100)
        
        if self.config_decoder.use_parallel:
            input_decoder_input_ids = decoder_input_ids[:, : -self.config_decoder.parallel_step]
            input_decoder_attention_mask = decoder_attention_mask[:, : -self.config_decoder.parallel_step]
        else:
            input_decoder_input_ids = decoder_input_ids[:, :-1]
            input_decoder_attention_mask = decoder_attention_mask[:, :-1]

        encoder_hidden_states = encoder_outputs[0]
        kwargs_decoder = {}
        if self.config_decoder.hidden_size != self.encoder_hidden_size:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        
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

        logits = decoder_outputs.logits
        return logits, labels

    def forward(self, inputs, targets=None):
        self.is_export = not self.training
        if not self.training:
            encoder_outputs = inputs
            model_kwargs = {
                "output_attentions": False,
                "output_hidden_states": False,
                "use_cache": True,
            }
            if self.is_export:
                word_pred = self.generate_export(encoder_outputs, model_kwargs)
            else:
                word_pred = self.generate(encoder_outputs, model_kwargs)
            return word_pred
            
        encoder_outputs, tgt_seq, mask = inputs
        logits, masked_labels = self.forwad_train(encoder_outputs, tgt_seq, mask)
        return logits, masked_labels