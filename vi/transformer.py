from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .base import VIBaseModule
from .priors import MeanFieldNormalPrior
from .utils.common_types import VIReturn, _prior_any_t, _vardist_any_t
from .variational_distributions import MeanFieldNormalVarDist


class VIMultiheadAttention(VIBaseModule):
    """Alpha implementation of VIMultiheadAttention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = True,
        variational_distribution: _vardist_any_t = MeanFieldNormalVarDist(),
        prior: _prior_any_t = MeanFieldNormalPrior(),
        prior_initialization: bool = False,
        rescale_prior: bool = True,
        return_log_probs: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        vikwargs = dict(
            variational_distribution=variational_distribution,
            prior=prior,
            prior_initialization=prior_initialization,
            rescale_prior=rescale_prior,
            return_log_probs=return_log_probs,
        )
        allkwargs = {**vikwargs, **factory_kwargs}

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = (self.kdim == embed_dim) and (self.vdim == embed_dim)

        self.num_heads = num_heads
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        variables: Dict[str, Tuple[int, ...]] = dict()
        random_variables: List[str] = []
        if not self._qkv_same_embed_dim:
            variables["q_proj_weight"] = (embed_dim, embed_dim)
            variables["k_proj_weight"] = (embed_dim, self.kdim)
            variables["v_proj_weight"] = (embed_dim, self.vdim)
            random_variables.extend(["q_proj_weight", "k_proj_weight", "v_proj_weight"])
        else:
            variables["in_proj_weight"] = (3 * embed_dim, embed_dim)
            random_variables.append("in_proj_weight")
        variables["out_proj_weight"] = (embed_dim, embed_dim)
        random_variables.append("out_proj_weight")
        if bias:
            variables["in_proj_bias"] = (3 * embed_dim,)
            variables["out_proj_bias"] = (embed_dim,)
            random_variables.extend(["in_proj_bias", "out_proj_bias"])

        if add_bias_kv:
            variables["bias_k"] = (1, 1, embed_dim)
            variables["bias_v"] = (1, 1, embed_dim)
            random_variables.extend(["bias_k", "bias_v"])

        self.random_variables = tuple(random_variables)
        self.add_kv_bias = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.bias = bias

        super().__init__(variables, **allkwargs)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        needs_weights: bool = True,
        key_padding_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> VIReturn[Tuple[Tensor, Optional[Tensor]]]:
        """Forward computation."""
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        # This transposition scheme is copied from torch including the comment
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        params = self.sample_variables()
        if self.add_kv_bias:
            bias_v = params.pop()
            bias_k = params.pop()
        else:
            bias_k = bias_v = None

        if not self._qkv_same_embed_dim:
            if self.bias:
                (
                    q_proj_weight,
                    k_proj_weight,
                    v_proj_weight,
                    out_proj_weight,
                    in_proj_bias,
                    out_proj_bias,
                ) = params
            else:
                q_proj_weight, k_proj_weight, v_proj_weight, out_proj_weight = params
                in_proj_bias = out_proj_bias = None
            in_proj_weight = None

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k=bias_k,
                bias_v=bias_v,
                add_zero_attn=self.add_zero_attn,
                dropout_p=0.0,
                out_proj_weight=out_proj_weight,
                out_proj_bias=out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=needs_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            if self.bias:
                in_proj_weight, out_proj_weight, in_proj_bias, out_proj_bias = params
            else:
                in_proj_weight, out_proj_weight = params
                in_proj_bias = out_proj_bias = None

            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                in_proj_weight,
                in_proj_bias,
                bias_k=bias_k,
                bias_v=bias_v,
                add_zero_attn=self.add_zero_attn,
                dropout_p=0.0,
                out_proj_weight=out_proj_weight,
                out_proj_bias=out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=needs_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        if self._return_log_probs:
            log_probs = self.get_log_probs(params)
            return (attn_output, attn_output_weights), log_probs
        else:
            return attn_output, attn_output_weights
