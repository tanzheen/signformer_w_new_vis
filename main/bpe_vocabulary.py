from tokenizers import ByteLevelBPETokenizer
from collections import Counter
from vocabulary import Vocabulary, TextVocabulary, UNK_TOKEN
from tqdm import tqdm
import sentencepiece as spm

def train_bpe_tokenizer(dataset, field, vocab_size=2800, min_frequency=2, save_path="bpe_tokenizer"):
    # Extract text from dataset and write to a temporary file
    texts = []
    for i in tqdm(dataset.values(), 
                    desc="Building vocabulary", 
                    unit="sample",
                    dynamic_ncols=True,  # Automatically adjust width
                    leave=True): 
        if field == "txt":
            texts.extend(i['text'].split())
        else:
            raise ValueError("Unknown field type")
  
    with open("temp_text.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(texts))
    
    spm.SentencePieceTrainer.train(
        input="temp_text.txt",
        model_prefix="spm_bpe",
        vocab_size=vocab_size,
        model_type="bpe",  # "bpe" to use Byte Pair Encoding
        character_coverage=1.0,  # Adjust based on your language
        unk_id=0,  # default UNK token ID
    )

    # Load the trained model
    sp = spm.SentencePieceProcessor()
    sp.load("spm_bpe.model")
    return sp

def build_vocab_bpe(
    field: str,  dataset, max_size: int, min_freq:int,  vocab_file: str = None
) -> Vocabulary:
    """
    Build a vocabulary using a BPE tokenizer.
    """

    tokenizer = train_bpe_tokenizer(dataset, field, vocab_size=max_size, min_frequency=min_freq)
    
    tokens = []
    for example in dataset.values():
        if field == "gls":
            text = " ".join(example['gloss'])
        elif field == "txt":
            text = " ".join(example['text'])
        else:
            raise ValueError("Unknown field type")
            
        # Tokenize the text using BPE
        encoding = tokenizer.encode(text)
        tokens.extend(encoding.tokens)
        
    # Count token frequencies
    counter = Counter(tokens)
    # Optionally, apply a frequency filter here if not already enforced in tokenizer training:
    # counter = {token: count for token, count in counter.items() if count >= min_frequency}
    
    # Sort tokens by frequency and then alphabetically, and cut off at max_size tokens.
    if min_freq > -1:
        counter = filter_min(counter, min_freq)
    vocab_tokens = sort_and_cut(counter, max_size)
    assert len(vocab_tokens) <= max_size


    if field == "txt":
        vocab = TextVocabulary(tokens=vocab_tokens)
    else:
        raise ValueError("Unknown vocabulary type")

    assert len(vocab) <= max_size + len(vocab.specials)
    assert vocab.itos[vocab.DEFAULT_UNK_ID()] == UNK_TOKEN

    for i, s in enumerate(vocab.specials):
        if i != vocab.DEFAULT_UNK_ID():
            assert not vocab.is_unk(s)

    return vocab

def filter_min(counter: Counter, minimum_freq: int):
    """ Filter counter by min frequency """
    filtered_counter = Counter({t: c for t, c in counter.items() if c >= minimum_freq})
    return filtered_counter


def sort_and_cut(counter: Counter, limit: int):
    """ Cut counter to most frequent,
    sorted numerically and alphabetically"""
    # sort by frequency, then alphabetically
    tokens_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
    tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    vocab_tokens = [i[0] for i in tokens_and_frequencies[:limit]]
    return vocab_tokens
