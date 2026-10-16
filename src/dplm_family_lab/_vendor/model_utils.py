# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Local selection: DPLM1 sampling helpers and minimal config types only.
from dataclasses import dataclass
import torch
@dataclass
class LoRAConfig:
    enable: bool = False
@dataclass
class NetConfig:
    arch_type: str = "esm"
    name: str = "airkingbd/dplm_150m"
def get_net(cfg):
    raise ValueError("Construct through the explicit pretrained backend; random initialization disabled.")
def get_net_class(name):
    from .esm import EsmForDPLM
    if name != "dplm_esm":
        raise ValueError(f"Unsupported DPLM type: {name}")
    return EsmForDPLM

def topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(scores) + 1e-8) + 1e-8
        )
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    masking = _scores < cutoff
    return masking


def topk_masking_prior(
    scores, cutoff_len, stochastic=False, temp=1.0, prior_mask=None
):
    """
    scores: [b, n]
    cutoff_len: [b, 1]
    stochastic: bool, whether to add noise to select top_k or not
    returns:
        mask: [b, n], with 1 if the token is in top-k lowest scores, 0 otherwise
    """
    if stochastic:
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(scores) + 1e-8) + 1e-8
        )
        _scores = scores + temp * gumbel_noise
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(
        dim=-1, index=cutoff_len
    )  # + torch.tensor(1e-10)
    # cutoff_len = k -> select k + 1 tokens
    masking = _scores < cutoff
    return masking


def mask_fill_811(inputs, masked_indices, mask_id):
    prev_tokens = inputs.clone()
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full_like(prev_tokens.float(), 0.8)).bool()
        & masked_indices
    )
    prev_tokens[indices_replaced] = mask_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full_like(prev_tokens.float(), 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(4, 24, prev_tokens.shape).type_as(prev_tokens)
    prev_tokens[indices_random] = random_words[indices_random]

    return prev_tokens


def sample_from_categorical(logits=None, temperature=1.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def stochastic_sample_from_categorical(
    logits=None, temperature=1.0, noise_scale=1.0
):
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(logits) + 1e-8) + 1e-8
    )
    logits = logits + noise_scale * gumbel_noise
    tokens, scores = sample_from_categorical(logits, temperature)
    # scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.95, filter_value=-float("Inf")
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    ori_shape = logits.shape
    logits = logits.reshape(-1, ori_shape[-1])
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (
            logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        )
        logits[indices_to_remove] = filter_value
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
        ..., :-1
    ].clone()
    sorted_indices_to_remove[..., 0] = 0
    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))
    logits = logits.reshape(ori_shape)
    return logits
