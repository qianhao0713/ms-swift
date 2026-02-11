from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Optional, Tuple, Union

import megatron.core
import torch
from megatron.core import tensor_parallel
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.gpt.gpt_layer_specs import TESpecProvider, get_gpt_layer_with_transformer_engine_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import (TransformerBlock, TransformerBlockSubmodules,
                                                         get_num_layers_to_build)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (TransformerLayer, TransformerLayerSubmodules,
                                                         get_transformer_layer_offset)
from megatron.core.utils import (WrappedTensor, deprecate_inference_params, make_viewless_tensor, nvtx_range_pop,
                                 nvtx_range_push)
from megatron.training import get_args
from packaging import version
from torch import Tensor

from swift.megatron.model.gpt_model import GPTModel
from swift.model import ModelType
from ..constant import MegatronModelType
from ..register import MegatronModelMeta, register_megatron_model
from .olmoe import OLMoEBridge, OLMoESelfAttention

mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')


class OLMo3SelfAttention(OLMoESelfAttention):

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, *args, **kwargs):
        _args = get_args()
        config_sw = deepcopy(config)
        layer_index = kwargs['layer_number'] - 1
        if _args.layer_types[layer_index] == 'sliding_attention':
            config_sw.window_size = _args.window_size
        elif _args.layer_types[layer_index] == 'full_attention':
            config_sw.window_size = None
        super().__init__(config_sw, submodules, *args, **kwargs)
        self.mscale = 1.0
        if _args.layer_types[layer_index] == 'sliding_attention':
            self.layer_type = 'sliding_attention'
        else:
            self.layer_type = 'full_attention'
            if isinstance(_args.rope_scaling, dict) and 'attention_factor' in _args.rope_scaling:
                self.mscale = _args.rope_scaling['attention_factor']

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> Tuple[Tensor, Tensor]:

        inference_context = deprecate_inference_params(inference_context, inference_params)

        is_inference_mode = inference_context is not None and not self.training

        is_using_flash_decode = is_inference_mode and self.config.flash_decode

        is_using_flashinfer_rope = is_inference_mode and (not inference_context.is_static_batching()
                                                          and inference_context.use_flashinfer_fused_rope)
        if is_using_flash_decode or is_using_flashinfer_rope:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb, ) * 2

        nvtx_range_push(suffix='qkv')
        split_qkv = True
        if self.attention_type != 'cross':
            assert not (self.config.fused_single_qkv_rope
                        and split_qkv), 'fused_single_qkv_rope requested but not available/supported for the config.'

        qkv_output = self.get_query_key_value_tensors(hidden_states, key_value_states, split_qkv=split_qkv)
        attn_mask_type = self.attn_mask_type
        query, key, value = qkv_output
        nvtx_range_pop(suffix='qkv')
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        nvtx_range_push(suffix='rotary_pos_emb')
        if rotary_pos_emb is not None and (not self.config.flash_decode or inference_context is None):
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    mscale=self.mscale,
                    cp_group=self.pg_collection.cp,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    mscale=self.mscale,
                    cp_group=self.pg_collection.cp,
                )

        nvtx_range_pop(suffix='rotary_pos_emb')

        nvtx_range_push(suffix='core_attention')
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix='core_attention')

        nvtx_range_push(suffix='linear_proj')
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix='linear_proj')

        return output, bias


@dataclass
class Olmo3TransformerLayerSubmodules(TransformerLayerSubmodules):
    post_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp


class TransformerLayerWithPostLayerNorm(TransformerLayer):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Olmo3TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config, submodules, layer_number, hidden_dropout, pg_collection, vp_stage)
        self.post_attn_layernorm = build_module(
            submodules.post_attn_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_mlp_layernorm = build_module(
            submodules.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

    def _forward_attention(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):
        residual = hidden_states
        nvtx_range_push(suffix='self_attention')
        attention_output_with_bias = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            rotary_pos_cos_sin=rotary_pos_cos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )
        nvtx_range_pop(suffix='self_attention')
        post_attn_layernorm_output = self.post_attn_layernorm(attention_output_with_bias[0])
        nvtx_range_push(suffix='self_attn_bda')
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                (post_attn_layernorm_output, None), residual, self.hidden_dropout)
        nvtx_range_pop(suffix='self_attn_bda')
        return hidden_states, context

    def _forward_mlp(self, hidden_states, inference_context=None):
        residual = hidden_states
        nvtx_range_push(suffix='mlp')
        if self.recompute_mlp:
            if self.config.fp8:
                # import here to avoid circular import
                from megatron.core.extensions.transformer_engine import te_checkpoint

                mlp_output_with_bias = te_checkpoint(
                    self.mlp,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    hidden_states,
                )
            else:
                mlp_output_with_bias = tensor_parallel.checkpoint(self.mlp, False, hidden_states)

        mlp_output_with_bias = self.mlp(hidden_states)

        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(mlp_output_with_bias[0])
        nvtx_range_pop(suffix='mlp')

        mlp_output, bias_output = mlp_output_with_bias
        mlp_output_layernorm = self.post_mlp_layernorm(mlp_output)
        mlp_output_with_bias = (mlp_output_layernorm, bias_output)

        nvtx_range_push(suffix='mlp_bda')
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(mlp_output_with_bias, residual,
                                                                                         self.hidden_dropout)
        nvtx_range_pop(suffix='mlp_bda')

        output = make_viewless_tensor(inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True)

        return output


class OLMo3Bridge(OLMoEBridge):

    def _set_layer_attn(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        mg_attn = None if mg_layer is None else mg_layer.self_attention
        hf_state_dict.update(self._set_attn_state(mg_attn, hf_state_dict, 'self_attn.', layer_idx, to_mcore))
        return hf_state_dict

    def _set_layer_mlp(self, mg_layer, hf_state_dict, layer_idx: int, to_mcore: bool):
        hf_mlp_prefix = 'mlp'
        mg_mlp = None if mg_layer is None else mg_layer.mlp
        hf_state_dict.update(self._set_mlp_state(mg_mlp, hf_state_dict, f'{hf_mlp_prefix}.', layer_idx, to_mcore))
        self._set_state_dict(mg_layer, 'post_attn_layernorm.weight', hf_state_dict, 'post_attention_layernorm.weight',
                             to_mcore)
        self._set_state_dict(mg_layer, 'post_mlp_layernorm.weight', hf_state_dict, 'post_feedforward_layernorm.weight',
                             to_mcore)
        return hf_state_dict


# rewrite this class to select sliding_rotary_pos_emb or full_rotaty_pos_emb in forward pass
class Olmo3TransformerBlock(TransformerBlock):

    def _checkpointed_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        sliding_rotary_pos_emb: Tensor,
        attention_bias: Tensor,
        packed_seq_params: PackedSeqParams,
        use_inner_quantization_context: bool,
    ):

        from megatron.core.transformer.transformer_block import te_checkpoint

        def custom(start: int, end: int):

            def custom_forward(hidden_states, attention_mask, context, context_mask, rotary_pos_emb,
                               sliding_rotary_pos_emb):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    if layer.self_attention.layer_type == 'sliding_attention':
                        layer_rotary_pos_emb = sliding_rotary_pos_emb
                    else:
                        layer_rotary_pos_emb = rotary_pos_emb
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()

                    with inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=layer_rotary_pos_emb,
                            attention_bias=attention_bias,
                            inference_context=None,
                            packed_seq_params=packed_seq_params,
                        )
                return hidden_states, context

            return custom_forward

        def checkpoint_handler(forward_func):
            if self.config.fp8 or self.config.fp4:
                return te_checkpoint(forward_func, self.config.distribute_saved_activations,
                                     tensor_parallel.random.get_cuda_rng_tracker, self.pg_collection.tp, hidden_states,
                                     attention_mask, context, context_mask, rotary_pos_emb, sliding_rotary_pos_emb)
            else:
                return tensor_parallel.checkpoint(forward_func, self.config.distribute_saved_activations, hidden_states,
                                                  attention_mask, context, context_mask, rotary_pos_emb,
                                                  sliding_rotary_pos_emb)

        if self.config.recompute_method == 'uniform':
            layer_idx = 0
            while layer_idx < self.num_layers_per_pipeline_rank:
                hidden_states, context = checkpoint_handler(
                    custom(layer_idx, layer_idx + self.config.recompute_num_layers))

                layer_idx += self.config.recompute_num_layers

        elif self.config.recompute_method == 'block':
            recompute_skip_num_layers = 0
            for layer_idx in range(self.num_layers_per_pipeline_rank):
                if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                    recompute_skip_num_layers += 1
                if (layer_idx >= recompute_skip_num_layers
                        and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers):
                    hidden_states, context = checkpoint_handler(custom(layer_idx, layer_idx + 1))
                else:
                    hidden_states, context = custom(layer_idx,
                                                    layer_idx + 1)(hidden_states, attention_mask, context, context_mask,
                                                                   rotary_pos_emb, sliding_rotary_pos_emb)
        else:
            raise ValueError('Invalid activation recompute method.')

        return hidden_states

    def forward(
        self,
        hidden_states: Union[Tensor, WrappedTensor],
        attention_mask: Optional[Tensor],
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        sliding_rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        rotary_pos_cos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        dynamic_inference_decode_only: Optional[bool] = None,
    ):

        inference_context = deprecate_inference_params(inference_context, inference_params)
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if self.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        if self.config.fp8:
            use_outer_quantization_context = self.config.fp8_recipe == Fp8Recipe.delayed
            use_inner_quantization_context = self.config.fp8_recipe != Fp8Recipe.delayed
            outer_quantization_context = (
                get_fp8_context(self.config) if use_outer_quantization_context else nullcontext())
        elif self.config.fp4:
            use_outer_quantization_context = False
            use_inner_quantization_context = True
            outer_quantization_context = nullcontext()
        else:
            use_outer_quantization_context = False
            use_inner_quantization_context = False
            outer_quantization_context = nullcontext()

        with rng_context, outer_quantization_context:
            if self.config.recompute_granularity == 'full' and self.training:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    sliding_rotary_pos_emb=sliding_rotary_pos_emb,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                    use_inner_quantization_context=use_inner_quantization_context,
                )
            else:
                for l_no, layer in enumerate(self.layers):
                    if use_inner_quantization_context:
                        if self.config.fp8:
                            inner_quantization_context = get_fp8_context(self.config, layer.layer_number - 1)
                        elif self.config.fp4:
                            inner_quantization_context = get_fp4_context(self.config, layer.layer_number - 1)
                        else:
                            inner_quantization_context = nullcontext()
                    else:
                        inner_quantization_context = nullcontext()
                    if layer.self_attention.layer_type == 'sliding_attention':
                        layer_rotary_pos_emb = sliding_rotary_pos_emb
                    else:
                        layer_rotary_pos_emb = rotary_pos_emb
                    with self.offload_context, inner_quantization_context:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=layer_rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            rotary_pos_cos_sin=rotary_pos_cos_sin,
                            attention_bias=attention_bias,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )

                    if (torch.is_grad_enabled() and self.config.cpu_offloading
                            and self.group_prefetch_offload_commit_async is not None):
                        hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

        # Final layer norm.
        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)
            hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

        if not self.pre_process and len(self.layers) == 0 and not self.final_layernorm:
            hidden_states = hidden_states.clone()

        return hidden_states


class Olmo3GPTModel(GPTModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotary_pos_emb_sliding = RotaryEmbedding(
            kv_channels=self.config.kv_channels,
            rotary_percent=kwargs['rotary_percent'],
            rotary_interleaved=self.config.rotary_interleaved,
            seq_len_interpolation_factor=kwargs['seq_len_interpolation_factor'],
            rotary_base=kwargs['rotary_base'],
            rope_scaling=kwargs['rope_scaling'],
            rope_scaling_factor=kwargs['rope_scaling_factor'],
            use_cpu_initialization=self.config.use_cpu_initialization,
            cp_group=self.pg_collection.cp,
        )
        self.decoder = Olmo3TransformerBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

    def _preprocess(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        decoder_input: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = \
            super()._preprocess(input_ids, position_ids, decoder_input, inference_context, packed_seq_params)
        rotary_seq_len = RotaryEmbedding.get_rotary_seq_len(self, inference_context, self.decoder, decoder_input,
                                                            self.config, packed_seq_params)
        packed_seq = packed_seq_params is not None and packed_seq_params.qkv_format == 'thd'
        slide_rotary_pos_emb = self.rotary_pos_emb_sliding(
            rotary_seq_len,
            packed_seq=packed_seq,
        )
        if packed_seq and not self.config.apply_rope_fusion:
            assert position_ids.shape[0] == 1, f'position_ids.shape: {position_ids.shape}'
            rotary_pos_emb = rotary_pos_emb[position_ids[0]]
        return decoder_input, rotary_pos_emb, slide_rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        decoder_input: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_context: BaseInferenceContext = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        inference_context = deprecate_inference_params(inference_context, inference_params)

        decoder_input, rotary_pos_emb, sliding_rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self._preprocess(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=decoder_input,
                inference_context=inference_context,
                packed_seq_params=packed_seq_params,
            ))
        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            sliding_rotary_pos_emb=sliding_rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
            **kwargs,
        )

        return self._postprocess(
            hidden_states=hidden_states,
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            loss_mask=loss_mask,
            decoder_input=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            runtime_gather_output=runtime_gather_output,
            extra_block_kwargs=extra_block_kwargs,
            inference_context=inference_context,
        )


def get_olmo3_decoder_block_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
) -> TransformerBlockSubmodules:
    layer_norm_impl = TENorm
    column_parallel_linear_impl = TESpecProvider().column_parallel_linear()
    kwargs = {'use_kitchen': config.use_kitchen} if mcore_013 else {}
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=True,
        multi_latent_attention=False,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        use_te_activation_func=True,
        **kwargs,
    )
    layer_specs = []
    for _ in range(config.num_layers):
        new_layer_spec = deepcopy(layer_spec)
        new_layer_spec.module = TransformerLayerWithPostLayerNorm
        kwargs = {}
        for f in fields(layer_spec.submodules):
            kwargs[f.name] = getattr(layer_spec.submodules, f.name)
        new_layer_spec.submodules = Olmo3TransformerLayerSubmodules(**kwargs)
        new_layer_spec.submodules.self_attention.module = OLMo3SelfAttention
        new_layer_spec.submodules.post_attn_layernorm = layer_norm_impl
        new_layer_spec.submodules.post_mlp_layernorm = layer_norm_impl
        new_layer_spec.submodules.post_attn_layernorm.module = layer_norm_impl
        new_layer_spec.submodules.post_mlp_layernorm.module = layer_norm_impl
        new_layer_spec.submodules.self_attention.submodules.linear_qkv = column_parallel_linear_impl
        new_layer_spec.submodules.mlp.submodules.linear_fc1 = column_parallel_linear_impl
        layer_specs.append(new_layer_spec)

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id] for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage)
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset:offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=layer_norm_impl)

    return block_spec


register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.olmo3,
        [ModelType.olmo3],
        get_transformer_layer_spec=get_olmo3_decoder_block_spec,
        model_cls=Olmo3GPTModel,
        bridge_cls=OLMo3Bridge,
    ))
