import torch
from pytest import raises
from torch import nn
from torch.nn import functional as F  # noqa: N812

from vi import (
    VIMultiheadAttention,
    VITransformerDecoderLayer,
    VITransformerEncoderLayer,
)
from vi.variational_distributions import MeanFieldNormalVarDist


def test_multiheadattention() -> None:
    """Test VIMultiheadAttention."""
    embed_dim = 5
    num_heads = 3
    with raises(AssertionError, match="embed_dim must be divisible by num_heads"):
        _ = VIMultiheadAttention(embed_dim, num_heads)

    embed_dim = 9
    random_variables1 = (
        "in_proj_weight",
        "out_proj_weight",
        "in_proj_bias",
        "out_proj_bias",
    )
    module1 = VIMultiheadAttention(
        embed_dim,
        num_heads,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
    )
    assert module1.embed_dim == embed_dim
    assert module1.num_heads == num_heads
    assert module1.kdim == embed_dim
    assert module1.vdim == embed_dim
    assert module1._qkv_same_embed_dim
    assert module1.batch_first
    assert module1.random_variables == random_variables1

    assert module1._in_proj_weight_mean.shape == (3 * embed_dim, embed_dim)
    assert module1._in_proj_weight_log_std.shape == (3 * embed_dim, embed_dim)
    assert module1._in_proj_bias_mean.shape == (3 * embed_dim,)
    assert module1._in_proj_bias_log_std.shape == (3 * embed_dim,)
    assert module1._out_proj_weight_mean.shape == (embed_dim, embed_dim)
    assert module1._out_proj_weight_log_std.shape == (embed_dim, embed_dim)
    assert module1._out_proj_bias_mean.shape == (embed_dim,)
    assert module1._out_proj_bias_log_std.shape == (embed_dim,)

    module1._has_sampling_responsibility = False

    src1 = torch.rand((3, 4, embed_dim))
    tgt1 = torch.rand((3, 5, embed_dim))
    extr = torch.rand((3, 5, embed_dim))

    tsrc1 = src1.transpose(0, 1)
    ttgt1 = tgt1.transpose(0, 1)

    # test q = k = v
    ref, attn_weights = F.multi_head_attention_forward(
        tsrc1,
        tsrc1,
        tsrc1,
        embed_dim,
        num_heads,
        module1._in_proj_weight_mean,
        module1._in_proj_bias_mean,
        None,
        None,
        False,
        0.0,
        module1._out_proj_weight_mean,
        module1._out_proj_bias_mean,
        average_attn_weights=False,
    )
    ref = ref.transpose(0, 1)
    (out, weights), lps = module1(src1, src1, src1, average_attn_weights=False)
    assert attn_weights.shape == weights.shape
    assert torch.allclose(attn_weights, weights)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)

    # test q != k = v
    ref, attn_weights = F.multi_head_attention_forward(
        tsrc1,
        ttgt1,
        ttgt1,
        embed_dim,
        num_heads,
        module1._in_proj_weight_mean,
        module1._in_proj_bias_mean,
        None,
        None,
        False,
        0.0,
        module1._out_proj_weight_mean,
        module1._out_proj_bias_mean,
        average_attn_weights=False,
    )
    ref = ref.transpose(0, 1)
    (out, weights), lps = module1(src1, tgt1, tgt1, average_attn_weights=False)
    assert attn_weights.shape == weights.shape
    assert torch.allclose(attn_weights, weights)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)

    # test q != k != v
    ref, attn_weights = F.multi_head_attention_forward(
        tsrc1,
        ttgt1,
        extr.transpose(0, 1),
        embed_dim,
        num_heads,
        module1._in_proj_weight_mean,
        module1._in_proj_bias_mean,
        None,
        None,
        False,
        0.0,
        module1._out_proj_weight_mean,
        module1._out_proj_bias_mean,
        average_attn_weights=False,
    )
    ref = ref.transpose(0, 1)
    (out, weights), lps = module1(src1, tgt1, extr, average_attn_weights=False)
    assert attn_weights.shape == weights.shape
    assert torch.allclose(attn_weights, weights)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)

    kdim = 3
    vdim = 5
    random_variables2 = (
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "out_proj_weight",
        "bias_k",
        "bias_v",
    )
    module2 = VIMultiheadAttention(
        embed_dim,
        num_heads,
        kdim=kdim,
        vdim=vdim,
        add_bias_kv=True,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        return_log_probs=False,
        batch_first=False,
        bias=False,
    )

    assert module2.embed_dim == embed_dim
    assert module2.num_heads == num_heads
    assert module2.kdim == kdim
    assert module2.vdim == vdim
    assert not module2._qkv_same_embed_dim
    assert not module2.batch_first
    assert module2.random_variables == random_variables2

    q = torch.rand((7, 4, embed_dim))
    k = torch.rand((5, 4, kdim))
    v = torch.rand((5, 4, vdim))

    ref, attn_weights = F.multi_head_attention_forward(
        q,
        k,
        v,
        embed_dim,
        num_heads,
        None,
        None,
        module2._bias_k_mean,
        module2._bias_v_mean,
        False,
        0.0,
        module2._out_proj_weight_mean,
        None,
        average_attn_weights=False,
        use_separate_proj_weight=True,
        q_proj_weight=module2._q_proj_weight_mean,
        k_proj_weight=module2._k_proj_weight_mean,
        v_proj_weight=module2._v_proj_weight_mean,
    )
    out, weights = module2(q, k, v, average_attn_weights=False)
    assert attn_weights.shape == weights.mean(dim=0).shape
    assert torch.allclose(attn_weights, weights.mean(dim=0), atol=2e-8)
    assert out.mean(dim=0).shape == ref.shape
    assert torch.allclose(out.mean(dim=0), ref)

    random_variables3 = (
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "out_proj_weight",
        "in_proj_bias",
        "out_proj_bias",
    )
    module3 = VIMultiheadAttention(
        embed_dim,
        num_heads,
        bias=True,
        kdim=kdim,
        vdim=vdim,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
    )
    module3._has_sampling_responsibility = False

    assert module3.embed_dim == embed_dim
    assert module3.num_heads == num_heads
    assert module3.kdim == kdim
    assert module3.vdim == vdim
    assert not module3._qkv_same_embed_dim
    assert module3.batch_first
    assert module3.random_variables == random_variables3

    q = torch.rand((4, 7, embed_dim))
    k = torch.rand((4, 5, kdim))
    v = torch.rand((4, 5, vdim))

    ref, attn_weights = F.multi_head_attention_forward(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        embed_dim,
        num_heads,
        None,
        module3._in_proj_bias_mean,
        None,
        None,
        False,
        0.0,
        module3._out_proj_weight_mean,
        module3._out_proj_bias_mean,
        average_attn_weights=False,
        use_separate_proj_weight=True,
        q_proj_weight=module3._q_proj_weight_mean,
        k_proj_weight=module3._k_proj_weight_mean,
        v_proj_weight=module3._v_proj_weight_mean,
    )
    (out, weights), _ = module3(q, k, v, average_attn_weights=False)
    ref = ref.transpose(0, 1)
    assert attn_weights.shape == weights.shape
    assert torch.allclose(attn_weights, weights)
    assert out.shape == ref.shape
    assert torch.allclose(out, ref)

    random_variables4 = ("in_proj_weight", "out_proj_weight")
    module4 = VIMultiheadAttention(
        embed_dim,
        num_heads,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        return_log_probs=False,
        batch_first=False,
        bias=False,
    )

    assert module4.embed_dim == embed_dim
    assert module4.num_heads == num_heads
    assert module4._qkv_same_embed_dim
    assert not module4.batch_first
    assert module4.random_variables == random_variables4

    src = torch.rand((7, 4, embed_dim))

    ref, attn_weights = F.multi_head_attention_forward(
        src,
        src,
        src,
        embed_dim,
        num_heads,
        module4._in_proj_weight_mean,
        None,
        None,
        None,
        False,
        0.0,
        module4._out_proj_weight_mean,
        None,
        average_attn_weights=True,
    )
    out, weights = module4(src, src, src, average_attn_weights=True)
    assert attn_weights.shape == weights.mean(dim=0).shape
    assert torch.allclose(attn_weights, weights.mean(dim=0), atol=2e-8)
    assert out.mean(dim=0).shape == ref.shape
    assert torch.allclose(out.mean(dim=0), ref)


def test_decoder_layer() -> None:
    """Test VITransformerDecoderLayer."""
    d_model = 8
    nhead = 2

    module1 = VITransformerDecoderLayer(d_model, nhead)
    assert not module1.norm_first
    assert module1.self_attn.embed_dim == d_model
    assert module1.self_attn.num_heads == nhead
    assert module1.self_attn.bias
    assert module1.self_attn.batch_first
    assert module1.multihead_attn.embed_dim == d_model
    assert module1.multihead_attn.num_heads == nhead
    assert module1.multihead_attn.bias
    assert module1.multihead_attn.batch_first
    assert isinstance(module1._ff_block[1], nn.ReLU)
    assert module1.norm1.eps == 1e-5
    assert module1.norm2.eps == 1e-5
    assert module1.norm3.eps == 1e-5
    assert module1.norm1.bias is not None
    assert module1.norm2.bias is not None
    assert module1.norm3.bias is not None

    module2 = VITransformerDecoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        activation=nn.GELU(),
        layer_norm_eps=1e-3,
        norm_first=True,
        bias=False,
        batch_first=False,
    )
    assert module2.norm_first
    assert module2.self_attn.embed_dim == d_model
    assert module2.self_attn.num_heads == nhead
    assert not module2.self_attn.bias
    assert not module2.self_attn.batch_first
    assert module2.multihead_attn.embed_dim == d_model
    assert module2.multihead_attn.num_heads == nhead
    assert not module2.multihead_attn.bias
    assert not module2.multihead_attn.batch_first
    assert isinstance(module2._ff_block[1], nn.GELU)
    assert module2.norm1.eps == 1e-3
    assert module2.norm2.eps == 1e-3
    assert module2.norm3.eps == 1e-3
    assert module2.norm1.bias is None
    assert module2.norm2.bias is None
    assert module2.norm3.bias is None

    module3 = VITransformerDecoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        norm_first=False,
        bias=False,
    )

    tgt = torch.rand((7, 4, d_model))
    mem = torch.rand((7, 4, d_model))

    (sa_out, sa_weights), sa_lps = module2._sa_block(tgt)
    (sa_ref, sa_rweight), sa_rlp = module2.self_attn(tgt, tgt, tgt)
    assert sa_out.shape == sa_ref.shape
    assert torch.allclose(sa_out, sa_ref)
    assert torch.allclose(sa_weights, sa_rweight)
    assert torch.allclose(sa_lps, sa_rlp)

    (mha_out, mha_weights), mha_lps = module2._mha_block(tgt, mem)
    (mha_ref, mha_rweight), mha_rlp = module2.multihead_attn(tgt, mem, mem)
    assert mha_out.shape == mha_ref.shape
    assert torch.allclose(mha_out, mha_ref)
    assert torch.allclose(mha_weights, mha_rweight)
    assert torch.allclose(mha_lps, mha_rlp)

    # check norm_first=True
    module2.return_log_probs(False)
    out1 = module2(tgt, mem)
    ref1 = tgt + module2._sa_block(module2.norm1(tgt))[0]
    ref1 = ref1 + module2._mha_block(module2.norm2(ref1), mem)[0]
    ref1 = module2._ff_block(module2.norm3(ref1))
    assert torch.allclose(out1, ref1)

    module2.return_log_probs(True)
    out2, _ = module2(tgt, mem)
    assert torch.allclose(out1, out2)

    # check norm_first=False
    module3.return_log_probs(False)
    out3 = module3(tgt, mem)
    ref2 = module3.norm1(tgt + module3._sa_block(tgt)[0])
    ref2 = module3.norm2(ref2 + module3._mha_block(ref2, mem)[0])
    ref2 = module3.norm3(module3._ff_block(ref2))
    assert torch.allclose(out3, ref2)

    module3.return_log_probs(True)
    out4, _ = module3(tgt, mem)
    assert torch.allclose(out3, out4)


def test_encoder_layer() -> None:
    """Test VITransformerEncoderLayer."""
    d_model = 8
    nhead = 2

    module1 = VITransformerEncoderLayer(d_model, nhead)
    assert not module1.norm_first
    assert module1.self_attn.embed_dim == d_model
    assert module1.self_attn.num_heads == nhead
    assert module1.self_attn.bias
    assert module1.self_attn.batch_first
    assert isinstance(module1._ff_block[1], nn.ReLU)
    assert module1.norm1.eps == 1e-5
    assert module1.norm2.eps == 1e-5
    assert module1.norm1.bias is not None
    assert module1.norm2.bias is not None

    module2 = VITransformerEncoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        activation=nn.GELU(),
        layer_norm_eps=1e-3,
        norm_first=True,
        bias=False,
        batch_first=False,
    )
    assert module2.norm_first
    assert module2.self_attn.embed_dim == d_model
    assert module2.self_attn.num_heads == nhead
    assert not module2.self_attn.bias
    assert not module2.self_attn.batch_first
    assert isinstance(module2._ff_block[1], nn.GELU)
    assert module2.norm1.eps == 1e-3
    assert module2.norm2.eps == 1e-3
    assert module2.norm1.bias is None
    assert module2.norm2.bias is None

    module3 = VITransformerEncoderLayer(
        d_model,
        nhead,
        variational_distribution=MeanFieldNormalVarDist(initial_std=1e-20),
        norm_first=False,
        bias=False,
    )

    src = torch.rand((7, 4, d_model))

    (sa_out, sa_weights), sa_lps = module2._sa_block(src)
    (sa_ref, sa_rweight), sa_rlp = module2.self_attn(src, src, src)
    assert sa_out.shape == sa_ref.shape
    assert torch.allclose(sa_out, sa_ref)
    assert torch.allclose(sa_weights, sa_rweight)
    assert torch.allclose(sa_lps, sa_rlp)

    # check norm_first=True
    module2.return_log_probs(False)
    out1 = module2(src)
    ref1 = src + module2._sa_block(module2.norm1(src))[0]
    ref1 = module2._ff_block(module2.norm2(ref1))
    assert torch.allclose(out1, ref1)

    module2.return_log_probs(True)
    out2, _ = module2(src)
    assert torch.allclose(out1, out2)

    # check norm_first=False
    module3.return_log_probs(False)
    out3 = module3(src)
    ref2 = module3.norm1(src + module3._sa_block(src)[0])
    ref2 = module3.norm2(module3._ff_block(ref2))
    assert torch.allclose(out3, ref2)

    module3.return_log_probs(True)
    out4, _ = module3(src)
    assert torch.allclose(out3, out4)
