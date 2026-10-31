"""Predecessor-conditioned low-rank corrections to a vocabulary shortlist."""

import torch

from .linear import UnquantizedProjection


class PreparedCandidateSelector:
    def __init__(self, hidden, weights, *, top_k, session, name):
        self.weights, self.top_k = weights, top_k
        self.projected = torch.zeros((*hidden.shape[:-1], weights["projection"].shape[0]),
                                     dtype=hidden.dtype, device=hidden.device)
        self.projection = UnquantizedProjection(hidden.flatten(0, -2), weights["projection"],
            self.projected.flatten(0, -2), session=session, name=name + ".hidden")

    def prepare(self, hidden):
        self.projection(hidden.flatten(0, -2))

    def __call__(self, logits, predecessor, position):
        unary, ids = logits.topk(self.top_k, dim=-1, sorted=False)
        coded = self.weights["predecessor"][predecessor] * self.projected[:, position]
        successors = self.weights["successor"][ids]
        # A per-row dot product over the rank; written as a multiply-and-sum so Inductor lowers it to one
        # fused kernel instead of autotuning a GEMM template for a [rows, top_k, rank] x [rows, rank, 1] product.
        correction = (successors.float() * coded.float()[:, None, :]).sum(-1)
        logits.fill_(-torch.inf)
        logits.scatter_(1, ids, unary + correction.float())
