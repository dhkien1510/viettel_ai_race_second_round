"""Minimal linear-chain CRF layer — no external dependency (no `pytorch-crf`).

Standard forward-algorithm negative log-likelihood (training) + Viterbi decode
(inference). batch_first convention throughout:
    emissions: (B, T, C) float
    tags:      (B, T)    long   (only positions where mask==1 are used)
    mask:      (B, T)    bool/float, RIGHT-padded — mask[:, 0] must be True for
               every example (real content first, padding trails at the end).
               This is what word-level pooling in crf_model.py produces.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        if num_tags < 1:
            raise ValueError("num_tags must be >= 1")
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood, averaged over the batch."""
        gold_score = self._score(emissions, tags, mask)
        log_partition = self._partition(emissions, mask)
        return (log_partition - gold_score).mean()

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi best path per example; each returned list has length equal
        to that example's real (mask==True) token count."""
        return self._viterbi(emissions, mask)

    def _score(self, emissions, tags, mask):
        batch_size, seq_len = tags.shape
        mask = mask.float()
        b = torch.arange(batch_size, device=emissions.device)
        score = self.start_transitions[tags[:, 0]] + emissions[b, 0, tags[:, 0]]
        for i in range(1, seq_len):
            score = score + self.transitions[tags[:, i - 1], tags[:, i]] * mask[:, i]
            score = score + emissions[b, i, tags[:, i]] * mask[:, i]
        seq_ends = mask.long().sum(1) - 1
        last_tags = tags[b, seq_ends]
        score = score + self.end_transitions[last_tags]
        return score

    def _partition(self, emissions, mask):
        _, seq_len, _ = emissions.shape
        mask = mask.float()
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        for i in range(1, seq_len):
            broadcast = (alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
                         + emissions[:, i].unsqueeze(1))
            new_alpha = torch.logsumexp(broadcast, dim=1)
            m = mask[:, i].unsqueeze(1)
            alpha = new_alpha * m + alpha * (1 - m)
        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def _viterbi(self, emissions, mask):
        batch_size, seq_len, _ = emissions.shape
        mask_bool = mask.bool()
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        history = []
        for i in range(1, seq_len):
            broadcast = (score.unsqueeze(2) + self.transitions.unsqueeze(0)
                         + emissions[:, i].unsqueeze(1))
            best_score, best_prev = broadcast.max(dim=1)
            m = mask_bool[:, i].unsqueeze(1)
            score = torch.where(m, best_score, score)
            history.append(best_prev)
        score = score + self.end_transitions.unsqueeze(0)
        seq_ends = mask_bool.long().sum(1) - 1

        best_tags_list: List[List[int]] = []
        for bi in range(batch_size):
            end = int(seq_ends[bi].item())
            best_last_tag = int(score[bi].argmax().item())
            best_tags = [best_last_tag]
            for hist in reversed(history[:end]):
                best_last_tag = int(hist[bi, best_tags[-1]].item())
                best_tags.append(best_last_tag)
            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list
