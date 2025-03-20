import os
import sentencepiece as spm
from collections import Counter
from tqdm import tqdm
from vocabulary import Vocabulary, TextVocabulary, UNK_TOKEN
from tokenizers import SentencePieceBPETokenizer
import transformers
class BPETokenizer:
    def __init__(self, vocab_size=4000, model_prefix="spm_bpe"):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        self.sp = None
        
    def train(self, dataset, field="txt", min_frequency=2):
        """
        Train a BPE tokenizer directly from dataset and load it into memory
        
        Args:
            dataset: Dictionary of data samples
            field: Field containing text to tokenize (default: "txt")
            min_frequency: Minimum frequency for tokens
        
        Returns:
            self: The trained tokenizer
        """
        # Extract text from dataset
        texts = []
        for sample in tqdm(dataset.values(), 
                           desc="Extracting text for tokenizer training", 
                           unit="sample",
                           dynamic_ncols=True):
            if field == "txt" or field == "bpe":
                texts.append(sample['text'])
            elif field == "gls":
                texts.append("".join(sample['gloss']))
            else:
                raise ValueError(f"Unknown field type: {field}")
        
        # Write text to temporary file for training
        temp_file = "temp_text.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        
        # Train the SentencePiece model
        tk_tokenizer = SentencePieceBPETokenizer()
        tk_tokenizer.train_from_iterator(
            texts,
            vocab_size=4000,
            min_frequency=2,
            show_progress=True,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<bos>", "<eos>"]
        )
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer, model_max_length=1024, special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<bos>", "<eos>"])
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
        print(f"bos_token_id: {tokenizer.bos_token_id}")
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
        print(f"pad_token_id: {tokenizer.pad_token_id}")
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
        print(f"eos_token_id: {tokenizer.eos_token_id}")
        tokenizer.unk_token = "<unk>"
        tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
        print(f"unk_token_id: {tokenizer.unk_token_id}")
        tokenizer.cls_token = "<cls>"
        tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
        print(f"cls_token_id: {tokenizer.cls_token_id}")
        tokenizer.sep_token = "<sep>"
        tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
        print(f"sep_token_id: {tokenizer.sep_token_id}")
        tokenizer.mask_token = "<mask>"
        tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
        print(f"mask_token_id: {tokenizer.mask_token_id}")

        # Load the trained model
        self.sp = tokenizer
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return self
    
    def encode(self, text):
        """
        Encode text to token IDs
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        print("encoding: ", text)
        return self.sp.batch_encode_plus(text, return_tensors="pt",padding = True, add_special_tokens=True )
    
    def decode(self, ids):
        """
        Decode token IDs to text
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.sp.batch_decode(ids, skip_special_tokens=False)
    
    def token_to_id(self, token):
        """
        Convert token to ID
        
        Args:
            token: Token string
            
        Returns:
            Token ID
        """
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.sp.piece_to_id(token)
    
    def id_to_token(self, id):
        """
        Convert ID to token
        
        Args:
            id: Token ID
            
        Returns:
            Token string
        """
        if self.sp is None:
            raise ValueError("Tokenizer not trained. Call train() first.")
        return self.sp.id_to_piece(id)
    


def create_bpe_tokenizer(dataset, field="txt", vocab_size=4000, min_frequency=2):
    """
    Convenience function to create and train a BPE tokenizer
    
    Args:
        dataset: Dictionary of data samples
        field: Field containing text to tokenize
        vocab_size: Size of vocabulary
        min_frequency: Minimum frequency threshold
        
    Returns:
        Trained BPETokenizer object
    """
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(dataset, field, min_frequency)
    print("length of vocab: ", len(tokenizer.sp.get_vocab()))
    # save the vocab
    with open(f"{tokenizer.model_prefix}.json", "w", encoding="utf-8") as f:
        for token, id in tokenizer.sp.get_vocab().items():
            f.write(f"{token} {id}\n")
    return tokenizer
