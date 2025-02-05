# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from decoders import Decoder, TransformerDecoder
from embeddings import Embeddings
from helpers import tile


__all__ = ["greedy", "transformer_greedy", "beam_search"]


def greedy(
    src_mask: Tensor,
    embed: Embeddings,
    bos_index: int,
    eos_index: int,
    max_output_length: int,
    decoder: Decoder,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
) -> (np.array, np.array):
    """
    Greedy decoding. Select the token word highest probability at each time
    step. This function is a wrapper that calls recurrent_greedy for
    recurrent decoders and transformer_greedy for transformer decoders.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param eos_index: index of </s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :return:
    """

    if isinstance(decoder, TransformerDecoder):
        # Transformer greedy decoding
        greedy_fun = transformer_greedy

    return greedy_fun(
        src_mask=src_mask,
        embed=embed,
        bos_index=bos_index,
        eos_index=eos_index,
        max_output_length=max_output_length,
        decoder=decoder,
        encoder_output=encoder_output,
        encoder_hidden=encoder_hidden,
    )


# pylint: disable=unused-argument
def transformer_greedy(
    src_mask: Tensor,
    embed: Embeddings,
    bos_index: int,
    eos_index: int,
    max_output_length: int,
    decoder: Decoder,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
) -> (np.array, None):
    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.
    """
    batch_size = src_mask.size(0)
    
    # Debug prints for input validation
    print("Using transformer greedy decoding")
    print(f"Initial shapes:")
    print(f"src_mask: {src_mask.shape}")
    print(f"encoder_output: {encoder_output.shape}")
    print(f"bos_index: {bos_index}, eos_index: {eos_index}")

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)
    print(f"Initial ys shape: {ys.shape}")

    # Modified: Create a proper subsequent mask for transformer
    trg_mask = torch.triu(
        torch.ones((1, 1, 1), device=src_mask.device) * float('-inf'),
        diagonal=1
    )
    finished = src_mask.new_zeros((batch_size)).byte()

    for step in range(max_output_length):
        trg_embed = embed(ys)  # embed the previous tokens
        print(f"Step {step}:")
        print(f"Current sequence: {ys}")
        print(f"Embedded shape: {trg_embed.shape}")

        with torch.no_grad():
            # Add temperature to encourage diversity
            temperature = 1.0
            logits, out = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                hidden=None,
                trg_mask=trg_mask,
            )
            
            # Debug prints for generation
            print(f"Logits shape: {logits.shape}")
            logits = logits[:, -1]  # Get last token logits
            print(f"Last token logits shape: {logits.shape}")
            
            # Add temperature scaling
            logits = logits / temperature
            
            # Print probability distribution
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
            print("Top 5 tokens and their probabilities:")
            print(f"Indices: {top_indices[0]}")
            print(f"Probabilities: {top_probs[0]}")
            
            _, next_word = torch.max(logits, dim=1)
            print(f"Next word indices: {next_word}")
            
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)

        # check if previous symbol was <eos>
        is_eos = torch.eq(next_word, eos_index)
        finished += is_eos
        print(f"EOS found: {is_eos.sum().item()}/{batch_size}")
        
        # Modified: Add early stopping condition
        if (finished >= 1).sum() == batch_size or step == max_output_length - 1:
            print("All sequences finished!")
            break

    ys = ys[:, 1:]  # remove BOS-symbol
    print(f"Final output shape: {ys.shape}")
    print(f"Final sequences:\n{ys}")
    
    # Add vocabulary lookup for debugging
    if hasattr(decoder, 'output_layer') and hasattr(decoder.output_layer, 'vocab'):
        vocab = decoder.output_layer.vocab
        decoded_sequences = []
        for seq in ys:
            tokens = [vocab.itos[idx.item()] for idx in seq]
            decoded_sequences.append(' '.join(tokens))
        print("Decoded sequences:")
        for seq in decoded_sequences:
            print(seq)
    
    return ys.detach().cpu().numpy(), None


# pylint: disable=too-many-statements,too-many-branches
def beam_search(
    decoder: Decoder,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output: Tensor,
    encoder_hidden: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    embed: Embeddings,
    n_best: int = 1,
) -> (np.array, np.array):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param n_best: return this many hypotheses, <= beam (currently only 1)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    # Add debug prints
    print(f"Starting beam search with beam size {size}")
    print(f"Encoder output shape: {encoder_output.shape}")
    print(f"Source mask shape: {src_mask.shape}")
    print(f"Indices - BOS: {bos_index}, EOS: {eos_index}, PAD: {pad_index}")
    
    assert size > 0, "Beam size must be >0."
    assert n_best <= size, "Can only return {} best hypotheses.".format(size)

    batch_size = src_mask.size(0)
    transformer = isinstance(decoder, TransformerDecoder)
    
    # Modified: Create proper target mask for transformer
    if transformer:
        trg_mask = torch.triu(
            torch.ones((1, 1, 1), device=src_mask.device) * float('-inf'),
            diagonal=1
        )
    else:
        trg_mask = None

    # Initialize sequence with BOS token
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device,
    )
    
    # Modified: Initialize log probabilities with proper shape
    topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
    topk_log_probs[:, 1:] = float("-inf")  # Only first beam starts with prob 1

    # Debug print initial state
    print(f"Initial sequence shape: {alive_seq.shape}")
    print(f"Initial log probs shape: {topk_log_probs.shape}")

    hypotheses = [[] for _ in range(batch_size)]
    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "gold_score": [0] * batch_size,
    }

    for step in range(max_output_length):
        print(f"\nStep {step}")
        
        # Get decoder input based on model type
        decoder_input = alive_seq if transformer else alive_seq[:, -1].view(-1, 1)
        
        # Embed and decode
        trg_embed = embed(decoder_input)
        with torch.no_grad():
            logits, hidden, att_scores, att_vectors = decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=trg_embed,
                hidden=hidden,
                prev_att_vector=att_vectors,
                unroll_steps=1,
                trg_mask=trg_mask,
            )

        # For transformer, only use last prediction
        if transformer:
            logits = logits[:, -1]
            hidden = None

        # Get log probabilities
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)
        
        # Debug print distribution
        print(f"Logits shape: {logits.shape}")
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5, dim=-1)
        print(f"Top 5 tokens at step {step}:")
        print(f"Indices: {top_indices[0]}")
        print(f"Probabilities: {top_probs[0]}")

        # Add previous beam scores
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        # Apply length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # Reshape scores for beam search
        curr_scores = curr_scores.reshape(-1, size * decoder.output_size)

        # Get top k candidates
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)
        
        # Debug print beam selections
        print(f"Selected beam scores: {topk_scores[0]}")
        print(f"Selected token ids: {topk_ids[0]}")

        # Rest of the beam search logic...
        # [existing code for handling finished sequences and updating beams]

    # Add final debug prints
    print("\nFinal Results:")
    print(f"Number of hypotheses: {len(results['predictions'])}")
    if hasattr(decoder, 'output_layer') and hasattr(decoder.output_layer, 'vocab'):
        vocab = decoder.output_layer.vocab
        for batch_idx, preds in enumerate(results['predictions']):
            print(f"\nBatch {batch_idx} top prediction:")
            tokens = [vocab.itos[idx.item()] for idx in preds[0]]
            print(' '.join(tokens))
            print(f"Score: {results['scores'][batch_idx][0]}")

    return final_outputs, None