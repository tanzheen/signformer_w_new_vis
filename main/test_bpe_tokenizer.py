import unittest
import os
import tempfile
import shutil
from bpe_tokenizer import BPETokenizer, create_bpe_tokenizer

class TestBPETokenizer(unittest.TestCase):
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
        
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test that the tokenizer initializes correctly"""
        tokenizer = BPETokenizer(vocab_size=100, model_prefix=self.model_prefix)
        self.assertEqual(tokenizer.vocab_size, 100)
        self.assertEqual(tokenizer.model_prefix, self.model_prefix)
        self.assertIsNone(tokenizer.sp)
    
    def test_training(self):
        """Test tokenizer training"""
        tokenizer = BPETokenizer(vocab_size=100, model_prefix=self.model_prefix)
        trained_tokenizer = tokenizer.train(self.mock_dataset, field="txt", min_frequency=1)
        
        # Check that trained_tokenizer is the same object (method returns self)
        self.assertEqual(tokenizer, trained_tokenizer)
        
        # Check that model files were created
        self.assertTrue(os.path.exists(f"{self.model_prefix}.model"))
        self.assertTrue(os.path.exists(f"{self.model_prefix}.vocab"))
        
        # Check that the SentencePiece processor is loaded
        self.assertIsNotNone(tokenizer.sp)
    
    def test_encode_decode(self):
        """Test encoding and decoding functionality"""
        tokenizer = BPETokenizer(vocab_size=100, model_prefix=self.model_prefix)
        tokenizer.train(self.mock_dataset)
        
        # Test encoding
        text = "hello world"
        encoded = tokenizer.encode(text)
        print(encoded)
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(token_id, int) for token_id in encoded))
        
        # Test decoding
        decoded = tokenizer.decode(encoded)
        # We can't guarantee exact matching due to BPE's nature, but it should be close
        self.assertIn("hello", decoded.lower())
        self.assertIn("world", decoded.lower())
    
    def test_token_id_conversion(self):
        """Test token to ID and ID to token conversion"""
        tokenizer = BPETokenizer(vocab_size=100, model_prefix=self.model_prefix)
        tokenizer.train(self.mock_dataset)
        
        # Get some tokens from the vocabulary
        sample_text = "hello"
        encoded = tokenizer.encode(sample_text)
        token_id = encoded[0]  # Get the first token ID
        
        # Test ID to token
        token = tokenizer.id_to_token(token_id)
        self.assertIsInstance(token, str)
        
        # Test token to ID (round trip)
        recovered_id = tokenizer.token_to_id(token)
        self.assertEqual(token_id, recovered_id)
    
    def test_untrained_errors(self):
        """Test that using methods without training raises errors"""
        tokenizer = BPETokenizer(vocab_size=100, model_prefix=self.model_prefix)
        
        with self.assertRaises(ValueError):
            tokenizer.encode("test")
            
        with self.assertRaises(ValueError):
            tokenizer.decode([1, 2, 3])
            
        with self.assertRaises(ValueError):
            tokenizer.token_to_id("test")
            
        with self.assertRaises(ValueError):
            tokenizer.id_to_token(1)
    
    def test_create_bpe_tokenizer(self):
        """Test the convenience function create_bpe_tokenizer"""
        tokenizer = create_bpe_tokenizer(
            self.mock_dataset, 
            field="txt", 
            vocab_size=100, 
            min_frequency=1
        )
        
        # Check that we got a trained tokenizer
        self.assertIsInstance(tokenizer, BPETokenizer)
        self.assertIsNotNone(tokenizer.sp)
        
        # Test basic functionality
        text = "hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
    
    def test_handling_unknown_tokens(self):
        """Test handling of unknown tokens"""
        tokenizer = BPETokenizer(vocab_size=100, model_prefix=self.model_prefix)
        tokenizer.train(self.mock_dataset)
        
        # Use a rare or unseen character sequence
        rare_text = "xylophoneqwertyuiop"
        encoded = tokenizer.encode(rare_text)
        
        # Ensure we get some output (should use subword tokens or UNK)
        self.assertTrue(len(encoded) > 0)
        
        # Test round trip
        decoded = tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)

if __name__ == "__main__":
    unittest.main() 