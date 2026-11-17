# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared variable-length attention with explicit compiler-visible storage."""

from functools import partial

import torch
from b12x.attention import varlen
from b12x.preparation import PreparedCall

from ignition.preparation.fixtures import varlen_attention as fixtures
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import declare, register_finalizer


class PreparedVarlenAttention:
    """Dense noncausal attention over packed segments.

    Queries and keys share `cu` unless `cu_k` gives the keys their own cumulative segment lengths, for
    segments whose keys outnumber their queries; `max_seqlen_k` then bounds one segment's keys.
    """

    # Without key segment lengths the keys follow the queries' lengths and bound.
    cu_k = None
    max_seqlen_k = None

    def __init__(self, q, k, v, cu, out, *, session, workspace, name, window=None, sink=None, max_seqlen=None,
                 math_mode="flash", cu_k=None, max_seqlen_k=None):
        self.q, self.k, self.v, self.cu, self.out, self.cu_k = q, k, v, cu, out, cu_k
        self.workspace = workspace
        self.sink = sink
        self.max_seqlen = q.shape[0] if max_seqlen is None else max_seqlen
        if type(self.max_seqlen) is not int or not 0 < self.max_seqlen <= q.shape[0]:
            raise ValueError("maximum attention sequence must fit the prepared query capacity")
        if cu_k is None and max_seqlen_k is not None:
            raise ValueError("a key segment bound requires key segment lengths")
        if cu_k is not None:
            self.max_seqlen_k = k.shape[0] if max_seqlen_k is None else max_seqlen_k
            if type(self.max_seqlen_k) is not int or not 0 < self.max_seqlen_k <= k.shape[0]:
                raise ValueError("maximum attention key sequence must fit the prepared key capacity")
        key_bound = self.max_seqlen_k or self.max_seqlen
        arithmetic = {} if math_mode == "flash" else {"math_mode": math_mode}
        self.plan = varlen.plan(q, k, v, cu, cu_k, max_seqlen_q=self.max_seqlen, max_seqlen_k=key_bound,
                               causal=False, window_size=window, attention_sink_bias=sink, **arithmetic)
        arguments = {
            "out": out,
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu,
            "max_seqlen_q": self.max_seqlen,
            "max_seqlen_k": key_bound,
            "softmax_scale": q.shape[-1] ** -0.5,
            "causal": False,
            "attention_sink_bias": sink,
        }
        if cu_k is not None:
            arguments["cu_seqlens_k"] = cu_k

        def prepare(state):
            scratch = planned_scratch(state.scratch_plan.scratch_specs(), workspace)
            lengths = {key: torch.zeros_like(value) for key, value in arguments.items() if key.startswith("cu_")}
            binding = state.bind(scratch=scratch, **{**arguments, **lengths})
            return PreparedCall(
                run=lambda: state.run(binding), owners=prime_owners(workspace, scratch, binding)
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=prepare,
                    benchmark_call=partial(fixtures.benchmark_call, arguments=arguments, workspace=workspace,
                        inputs=fixtures.input_factory(arguments, storage=getattr(session, "benchmark_storage", None),
                                                      session=session)),
                ),
            ),
        )
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def functional(self, *, session):
        self.output_layout = (tuple(self.out.shape), tuple(self.out.stride()), self.out.dtype)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.q = self.k = self.v = self.cu = self.cu_k = self.out = self.workspace = self.scratch = None

    def __call__(self, q=None, k=None, v=None, cu=None, cu_k=None):
        if self.out is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=q.device).zero_()
            scratch, = (torch.empty(shape, dtype=dtype, device=q.device) for shape, dtype in self.scratch_layout)
        else:
            q, k, v, cu, cu_k, output, scratch = self.q, self.k, self.v, self.cu, self.cu_k, self.out, self.scratch
        varlen.run_out(q, k, v, cu, out=output, cu_seqlens_k=cu_k,
            scratch=scratch, plan=self.plan, attention_sink_bias=self.sink,
            max_seqlen_q=self.max_seqlen, max_seqlen_k=self.max_seqlen_k or self.max_seqlen)
        return output
