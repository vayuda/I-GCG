import gc

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_attacks import get_embedding_matrix, get_embeddings

class UCBTracker:
    def __init__(self, device):
        self.token_counts = {}  # (position, token_id) -> count
        self.total_pulls = 1
        self.device = device

    def get_count(self, position, token_id):
        return self.token_counts.get((position.item(), token_id.item()), 1)

    def update_count(self, position, token_id):
        key = (position.item(), token_id.item())
        self.token_counts[key] = self.token_counts.get(key, 1) + 1


def sample_control_ucb(control_toks, grad, batch_size, topk=256, not_allowed_tokens=None, ucb_tracker=None):
    """
    Sample control tokens using UCB strategy
    UCB = Q(a) + c * sqrt(ln(t)/N(a))
    """
    if ucb_tracker is None:
        ucb_tracker = UCBTracker(grad.device)

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    # Calculate UCB scores for each position
    c = 2.0  # exploration coefficient
    ucb_scores = torch.zeros_like(grad)

    # Calculate UCB scores for each position
    for pos in range(grad.shape[0]):
        exploitation_term = -grad[pos]  # negative gradient as immediate reward

        # Calculate exploration terms using stored counts
        counts = torch.tensor([ucb_tracker.get_count(torch.tensor(pos), torch.tensor(token_id))
                             for token_id in range(grad.shape[1])],
                             device=grad.device)

        exploration_term = c * torch.sqrt(
            torch.log(torch.tensor(ucb_tracker.total_pulls, device=grad.device)) / counts
        )
        ucb_scores[pos] = exploitation_term + exploration_term

    # Get top k indices based on UCB scores
    top_indices = ucb_scores.topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)

    # Select tokens with highest UCB scores
    new_token_val = top_indices[new_token_pos, 0].unsqueeze(-1)

    # Update counts for selected tokens
    for pos, token in zip(new_token_pos, new_token_val):
        ucb_tracker.update_count(pos, token)
    ucb_tracker.total_pulls += 1

    # Create new candidates
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.

    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    with torch.enable_grad():
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        # now stitch it together with the rest of the embeddings
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:,:input_slice.start,:],
                input_embeds,
                embeds[:,input_slice.stop:,:]
            ],
            dim=1)

        logits = model(inputs_embeds=full_embeds).logits
        targets = input_ids[target_slice]
        loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], targets)

        loss.backward()

        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)

    return grad

def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None):

    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0,
        len(control_toks),
        len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1),
        device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    cands = []
    target_length = len(control_cand[0])  # Length we want to maintain

    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        encoded_ids = tokenizer(decoded_str, add_special_tokens=False).input_ids

        # If too long, truncate
        if len(encoded_ids) > target_length:
            truncated_str = tokenizer.decode(encoded_ids[:target_length], skip_special_tokens=True)
            encoded_ids = tokenizer(truncated_str, add_special_tokens=False).input_ids

        # If too short, pad with "!"
        if len(encoded_ids) < target_length:
            padding_needed = target_length - len(encoded_ids)
            decoded_str = decoded_str + " " + "!" * padding_needed
            encoded_ids = tokenizer(decoded_str, add_special_tokens=False).input_ids

        # Verify length is correct after modification
        if len(encoded_ids) == target_length:
            cands.append(decoded_str)
        else:
            # Fallback to current control if we still can't get correct length
            cands.append(curr_control)

    # Ensure we have enough candidates
    while len(cands) < len(control_cand):
        cands.append(cands[0])
    return cands


def get_logits(*, model, tokenizer, input_ids, control_slice, test_controls=None, return_ids=False, batch_size=512):

    if isinstance(test_controls[0], str):
        max_len = control_slice.stop - control_slice.start
        test_ids = [
            torch.tensor(tokenizer(control, add_special_tokens=False).input_ids[:max_len], device=model.device)
            for control in test_controls
        ]
        pad_tok = 0
        while pad_tok in input_ids or any([pad_tok in ids for ids in test_ids]):
            pad_tok += 1
        nested_ids = torch.nested.nested_tensor(test_ids)
        test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
    else:
        raise ValueError(f"test_controls must be a list of strings, got {type(test_controls)}")

    if not(test_ids[0].shape[0] == control_slice.stop - control_slice.start):
        raise ValueError((
            f"test_controls must have shape "
            f"(n, {control_slice.stop - control_slice.start}), "
            f"got {test_ids.shape}"
        ))

    locs = torch.arange(control_slice.start, control_slice.stop).repeat(test_ids.shape[0], 1).to(model.device)
    ids = torch.scatter(
        input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
        1,
        locs,
        test_ids
    )
    if pad_tok >= 0:
        attn_mask = (ids != pad_tok).type(ids.dtype)
    else:
        attn_mask = None

    if return_ids:
        del locs, test_ids ; gc.collect()
        return forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size), ids
    else:
        del locs, test_ids
        logits = forward(model=model, input_ids=ids, attention_mask=attn_mask, batch_size=batch_size)
        del ids ; gc.collect()
        return logits


def forward(*, model, input_ids, attention_mask, batch_size=512):

    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i+batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i+batch_size]
        else:
            batch_attention_mask = None

        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)

def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start-1, target_slice.stop-1)
    loss = crit(logits[:,loss_slice,:].transpose(1,2), ids[:,target_slice])
    return loss.mean(dim=-1)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            **kwargs
        ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
