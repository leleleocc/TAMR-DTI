# Copyright (c) 2024, Tri Dao.

from typing import Optional, Tuple

import torch

import causal_conv1d_cuda


LIBRARY_NAME = "DaoAILab"


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_fwd_cpp", mutates_args={"out", "final_states_out"})
def _causal_conv1d_fwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    out: torch.Tensor,
    final_states_out: Optional[torch.Tensor],
    silu_activation: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_fwd(
        x,
        weight,
        bias,
        seq_idx,
        initial_states,
        out,
        final_states_out,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_bwd_cpp", mutates_args={
    "dfinal_states",
    "dx",
    "dweight",
    "dbias",
    "dinitial_states",
})
def _causal_conv1d_bwd_cpp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    dfinal_states: Optional[torch.Tensor],
    dx: torch.Tensor,
    dweight: torch.Tensor,
    dbias: Optional[torch.Tensor],
    dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> None:
    causal_conv1d_cuda.causal_conv1d_bwd(
        x,
        weight,
        bias,
        dout,
        seq_idx,
        initial_states,
        dfinal_states,
        dx,
        dweight,
        dbias,
        dinitial_states,
        silu_activation,
    )


@torch.library.custom_op(f"{LIBRARY_NAME}::_causal_conv1d_update_cpp", mutates_args={"out", "conv_state"})
def _causal_conv1d_update_cpp(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
) -> None:
    causal_conv1d_cuda.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        out,
        silu_activation,
        cache_seqlens,
        conv_state_indices
    )


def causal_conv1d_fwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    final_states_out: Optional[torch.Tensor],
    silu_activation: bool,
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_fwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        seq_idx=seq_idx,
        initial_states=initial_states,
        out=out,
        final_states_out=final_states_out,
        silu_activation=silu_activation,
    )
    return out


def causal_conv1d_bwd_function(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    dout: torch.Tensor,
    seq_idx: Optional[torch.Tensor],
    initial_states: Optional[torch.Tensor],
    dfinal_states: Optional[torch.Tensor],
    dx: Optional[torch.Tensor],
    return_dinitial_states: torch.Tensor,
    silu_activation: bool,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    batch_size, dim = x.size()[:2]
    width = weight.size(-1)

    if dx is None:
        dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight, dtype=torch.float32)
    dbias = None
    if bias is not None:
        dbias = torch.zeros_like(bias, dtype=torch.float32)
    dinitial_states = None
    if return_dinitial_states:
        dinitial_states = torch.empty(batch_size, width - 1, dim, device=x.device, dtype=x.dtype).transpose(1, 2)

    _causal_conv1d_bwd_cpp(
        x=x,
        weight=weight,
        bias=bias,
        dout=dout,
        seq_idx=seq_idx,
        initial_states=initial_states,
        dfinal_states=dfinal_states,
        dx=dx,
        dweight=dweight,
        dbias=dbias,
        dinitial_states=dinitial_states,
        silu_activation=silu_activation,
    )

    dweight = dweight.type_as(weight)
    if dbias is not None:
        dbias = dbias.type_as(bias)
    return dx, dweight, dbias, dinitial_states


def causal_conv1d_update_function(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    silu_activation: bool,
    cache_seqlens: Optional[torch.Tensor],
    conv_state_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    out = torch.empty_like(x)
    _causal_conv1d_update_cpp(
        x=x,
        conv_state=conv_state,
        weight=weight,
        bias=bias,
        out=out,
        silu_activation=silu_activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=conv_state_indices,
    )
    return out
