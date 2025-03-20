#!/usr/bin/env python
"""
Example script demonstrating how to use the BPETokenizer class with a small dataset.
"""
import os
from bpe_tokenizer import BPETokenizer
from pprint import pprint

def main():
    # Create a small example dataset
    example_dataset = {
        '1': {'text': 'Hello world! This is a test.', 'gloss': ['hello', 'world', 'test']},
        '2': {'text': 'Natural language processing with BPE tokenization.', 'gloss': ['natural', 'language', 'processing', 'bpe', 'tokenization']},
        '3': {'text': 'Byte-pair encoding is a simple form of data compression.', 'gloss': ['byte-pair', 'encoding', 'data', 'compression']},
        '4': {'text': 'Tokenization splits text into smaller units called tokens.', 'gloss': ['tokenization', 'text', 'tokens']},
        '5': {'text': 'BPE iteratively merges the most frequent pairs of bytes or characters.', 'gloss': ['bpe', 'merges', 'frequent', 'pairs']},
    }
    
    print("Creating and training BPE tokenizer with vocabulary size of 100...")
    # Create and train the tokenizer (low vocab size for this example)
    tokenizer = BPETokenizer(vocab_size=100, model_prefix="example_bpe")
    tokenizer.train(example_dataset)
    
    # Test sentences
    test_sentences = [
        "Hello world!",
        "This is an example sentence.",
        "BPE tokenization works on subword units.",
        "Unseen words can be broken down into smaller known subwords."
    ]
    
    print("\nTokenizing example sentences:")
    for i, sentence in enumerate(test_sentences):
        print(f"\nExample {i+1}: '{sentence}'")
        
        # Encode the sentence
        token_ids = tokenizer.encode(sentence)
        print(f"Token IDs: {token_ids}")
        
        # Decode back to text
        reconstructed = tokenizer.decode(token_ids)
        print(f"Reconstructed: '{reconstructed}'")
        
        # Show individual tokens
        tokens = [tokenizer.id_to_token(id) for id in token_ids]
        print(f"Tokens: {tokens}")
    
    # Demonstrate handling of unknown tokens
    print("\nHandling unknown or rare tokens:")
    rare_sentence = "xylophone pneumonoultramicroscopicsilicovolcanoconiosis"
    token_ids = tokenizer.encode(rare_sentence)
    tokens = [tokenizer.id_to_token(id) for id in token_ids]
    print(f"Rare sentence: '{rare_sentence}'")
    print(f"Tokenized as: {tokens}")
    print(f"Token IDs: {token_ids}")
    reconstructed = tokenizer.decode(token_ids)
    print(f"Reconstructed: '{reconstructed}'")
    
    # Show a portion of the vocabulary
    print("\nPart of the BPE vocabulary:")
    vocab = {tokenizer.id_to_token(i): i for i in range(min(20, tokenizer.vocab_size))}
    pprint(vocab)
    
    # Clean up model files
    print("\nCleaning up model files...")
    if os.path.exists("example_bpe.model"):
        os.remove("example_bpe.model")
    if os.path.exists("example_bpe.vocab"):
        os.remove("example_bpe.vocab")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 