import unittest
import os
import tempfile
import shutil
from bpe_tokenizer import BPETokenizer
from vocabulary import TextVocabulary

class TestBPEVocabularyIntegration(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test artifacts
        self.test_dir = tempfile.mkdtemp()
        
        # Mock dataset for testing
        self.mock_dataset = {
            '1': {'text': 'hello world', 'gloss': ['hello', 'world']},
            '2': {'text': 'testing the tokenizer', 'gloss': ['testing', 'the', 'tokenizer']},
            '3': {'text': 'this is a longer sentence with more tokens', 'gloss': ['this', 'is', 'a', 'longer', 'sentence', 'with', 'more', 'tokens']},
            '4': {'text': 'repeat repeat repeat token token', 'gloss': ['repeat', 'repeat', 'repeat', 'token', 'token']},
            '5': {'text': 'special characters: !@#$%^&*()', 'gloss': ['special', 'characters', '!@#$%^&*()']},
        }
        
        # Set model prefix to be in the test directory
        self.model_prefix = os.path.join(self.test_dir, "test_bpe")
        
        # Create and train tokenizer once for all tests
        self.tokenizer = BPETokenizer(vocab_size=100, model_prefix=self.model_prefix)
        self.tokenizer.train(self.mock_dataset)
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_get_vocab_method(self):
        """Test adding the get_vocab method to BPETokenizer"""
        # First add the get_vocab method to the BPETokenizer class if it's not already there
        if not hasattr(BPETokenizer, 'get_vocab'):
            BPETokenizer.get_vocab = lambda self: {
                self.sp.id_to_piece(i): i 
                for i in range(self.sp.get_piece_size())
            }
        
        # Test the method
        vocab_dict = self.tokenizer.get_vocab()
        self.assertIsInstance(vocab_dict, dict)
        self.assertTrue(len(vocab_dict) > 0)
        
        # Check that keys are strings and values are integers
        for token, idx in vocab_dict.items():
            self.assertIsInstance(token, str)
            self.assertIsInstance(idx, int)
    
    def test_build_vocabulary_method(self):
        """Test adding build_vocabulary method to BPETokenizer"""
        # First add the build_vocabulary method to the BPETokenizer class if it's not already there
        if not hasattr(BPETokenizer, 'build_vocabulary'):
            from collections import Counter
            from tqdm import tqdm
            
            def build_vocabulary(self, dataset, field="txt", min_freq=2):
                # Collect all tokens from the dataset using the trained tokenizer
                tokens = []
                for sample in tqdm(dataset.values(), 
                                desc="Building vocabulary", 
                                unit="sample",
                                dynamic_ncols=True):
                    if field == "txt" or field == "bpe":
                        text = sample['text']
                    elif field == "gls":
                        text = "".join(sample['gloss'])
                    else:
                        raise ValueError(f"Unknown field type: {field}")
                    
                    # Tokenize and add to token list
                    token_ids = self.encode(text)
                    decoded_tokens = [self.id_to_token(id) for id in token_ids]
                    tokens.extend(decoded_tokens)
                
                # Count and filter tokens
                counter = Counter(tokens)
                if min_freq > 1:
                    counter = Counter({t: c for t, c in counter.items() if c >= min_freq})
                
                # Sort by frequency, then alphabetically
                tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
                tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
                
                # Create the vocabulary
                vocab_tokens = [i[0] for i in tokens_and_frequencies[:self.vocab_size]]
                vocab = TextVocabulary(tokens=vocab_tokens)
                
                return vocab
                
            BPETokenizer.build_vocabulary = build_vocabulary
        
        # Test the method
        vocab = self.tokenizer.build_vocabulary(self.mock_dataset, min_freq=1)
        
        # Check that we got a TextVocabulary
        self.assertIsInstance(vocab, TextVocabulary)
        
        # Check that the vocabulary has items
        self.assertTrue(len(vocab) > 0)
        
        # Test using the vocabulary
        # Get a token from our mock dataset
        test_text = "hello"
        # Encode with the tokenizer
        token_ids = self.tokenizer.encode(test_text)
        # Convert IDs to tokens
        tokens = [self.tokenizer.id_to_token(id) for id in token_ids]
        
        # Try to look up in vocabulary
        for token in tokens:
            # Either the token is in vocabulary or it's mapped to UNK
            if token in vocab.stoi:
                idx = vocab.stoi[token]
                self.assertIsInstance(idx, int)
                # Check round-trip conversion
                self.assertEqual(vocab.itos[idx], token)
    
    def test_end_to_end_tokenization_workflow(self):
        """Test complete tokenization workflow from raw text to vocabulary"""
        # Train the tokenizer (already done in setUp)
        # Create vocabulary
        vocab = self.tokenizer.build_vocabulary(self.mock_dataset, min_freq=1)
        
        # Test the complete workflow with a sample sentence
        test_sentence = "hello world tokenizer test"
        
        # 1. Tokenize with BPE
        token_ids = self.tokenizer.encode(test_sentence)
        bpe_tokens = [self.tokenizer.id_to_token(id) for id in token_ids]
        
        # 2. Convert tokens to indices in our vocabulary
        indices = []
        for token in bpe_tokens:
            if token in vocab.stoi:
                indices.append(vocab.stoi[token])
            else:
                indices.append(vocab.stoi[vocab.UNK_TOKEN])
        
        # 3. Convert indices back to tokens
        recovered_tokens = [vocab.itos[idx] for idx in indices]
        
        # Verify we can recover our tokens
        self.assertEqual(len(bpe_tokens), len(recovered_tokens))
        
        # Tokens should match or be UNK
        for original, recovered in zip(bpe_tokens, recovered_tokens):
            self.assertTrue(
                original == recovered or recovered == vocab.UNK_TOKEN
            )

if __name__ == "__main__":
    unittest.main() 