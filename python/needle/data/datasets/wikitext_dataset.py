import os

import numpy as np
from needle import backend_ndarray as nd
from needle import Tensor


from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """
    Simple BPE subword tokenizer.

    - Train on an iterator of text lines.
    - Builds a subword vocab up to `vocab_size` using BPE merges.
    - Encodes text into lists of token IDs.
    """

    def __init__(self,
                 vocab_size: int = 30000,
                 min_freq: int = 2,
                 specials: Optional[List[str]] = None):
        if specials is None:
            specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        self._target_vocab_size = vocab_size
        self.min_freq = min_freq
        self.specials = specials

        self.token2idx: Dict[str, int] = {}
        self.idx2token: List[str] = []
        self.merges: List[Tuple[str, str]] = []
        self.trained = False

        for s in self.specials:
            self._add_token(s)

    def _add_token(self, tok: str):
        if tok not in self.token2idx:
            idx = len(self.idx2token)
            self.idx2token.append(tok)
            self.token2idx[tok] = idx

    def _word_to_symbols(self, word: str) -> List[str]:
        # characters plus end-of-word marker
        return list(word) + ["</w>"]

    def _init_base_vocab(self, word_freqs: Dict[str, int]):
        chars = set()
        for w in word_freqs.keys():
            for ch in w:
                chars.add(ch)
        chars.add("</w>")
        for ch in sorted(chars):
            self._add_token(ch)

    def _train_from_word_freqs(self, word_freqs: Dict[str, int]):
        # initialize char-level vocab
        self._init_base_vocab(word_freqs)
        word_symbols = {w: self._word_to_symbols(w) for w in word_freqs.keys()}

        def get_pair_stats():
            pair_counts = defaultdict(int)
            for w, freq in word_freqs.items():
                syms = word_symbols[w]
                if len(syms) < 2:
                    continue
                for i in range(len(syms) - 1):
                    pair = (syms[i], syms[i + 1])
                    pair_counts[pair] += freq
            return pair_counts

        target_size = self._target_vocab_size
        while len(self.idx2token) < target_size:
            pair_counts = get_pair_stats()
            if not pair_counts:
                break
            (best_pair, best_freq) = max(pair_counts.items(), key=lambda x: x[1])
            if best_freq < self.min_freq:
                break

            left, right = best_pair
            new_sym = left + right
            self._add_token(new_sym)
            self.merges.append(best_pair)

            # replace pair in all words
            for w in word_symbols.keys():
                syms = word_symbols[w]
                new_syms = []
                i = 0
                while i < len(syms):
                    if (
                        i < len(syms) - 1
                        and syms[i] == left
                        and syms[i + 1] == right
                    ):
                        new_syms.append(new_sym)
                        i += 2
                    else:
                        new_syms.append(syms[i])
                        i += 1
                word_symbols[w] = new_syms

            if len(self.idx2token) >= target_size:
                break

    def train_from_iterator(self, text_iter):
        """
        Train BPE from an iterator over lines of text (e.g., the train file).
        """
        word_freqs = Counter()
        for line in text_iter:
            line = line.strip()
            if not line:
                continue
            for w in line.split():
                word_freqs[w] += 1

        self._train_from_word_freqs(word_freqs)
        self.trained = True

    def _encode_word_symbols(self, word: str) -> List[str]:
        syms = self._word_to_symbols(word)
        # apply merges greedily in learned order
        for (left, right) in self.merges:
            i = 0
            new_syms = []
            while i < len(syms):
                if (
                    i < len(syms) - 1
                    and syms[i] == left
                    and syms[i + 1] == right
                ):
                    new_syms.append(left + right)
                    i += 2
                else:
                    new_syms.append(syms[i])
                    i += 1
            syms = new_syms
        return syms

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        assert self.trained, "Call train_from_iterator() before encode()."
        ids = []
        if add_bos and "<bos>" in self.token2idx:
            ids.append(self.token2idx["<bos>"])
        for w in text.strip().split():
            syms = self._encode_word_symbols(w)
            for s in syms:
                tok_id = self.token2idx.get(s, self.token2idx.get("<unk>", 0))
                ids.append(tok_id)
        if add_eos and "<eos>" in self.token2idx:
            ids.append(self.token2idx["<eos>"])
        return ids

    def __len__(self):
        return len(self.idx2token)


class Dictionary(object):
    """
    Word-level dictionary (kept for backward compatibility).
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.idx2word)


def _iter_lines(path, max_lines=None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            yield line


class Corpus(object):
    """
    Creates corpus from WikiText-103 files.

    If use_subword=False (default):
        - uses word-level Dictionary (legacy behavior)
    If use_subword=True:
        - trains a BPETokenizer on wiki.train.tokens
        - encodes train/valid/test into subword IDs
    """
    def __init__(self,
                 base_dir,
                 max_lines=None,
                 use_subword: bool = False,
                 subword_vocab_size: int = 30000,
                 subword_min_freq: int = 2):
        self.use_subword = use_subword

        train_path = os.path.join(base_dir, "wiki.train.tokens")
        valid_path = os.path.join(base_dir, "wiki.valid.tokens")
        test_path  = os.path.join(base_dir, "wiki.test.tokens")

        if not use_subword:
            # -------- original word-level behavior --------
            self.dictionary = Dictionary()
            self.train = self._tokenize_word_level(train_path, max_lines)
            self.valid = self._tokenize_word_level(valid_path, max_lines)
            self.test  = self._tokenize_word_level(test_path,  max_lines)
        else:
            # -------- subword / BPE behavior --------
            self.tokenizer = BPETokenizer(
                vocab_size=subword_vocab_size,
                min_freq=subword_min_freq,
            )
            # Train BPE on train file, streaming
            self.tokenizer.train_from_iterator(_iter_lines(train_path, max_lines))

            # Encode splits into lists of IDs
            self.train = self._encode_file(train_path, max_lines)
            self.valid = self._encode_file(valid_path, max_lines)
            self.test  = self._encode_file(test_path,  max_lines)

    # ----- word-level path -----
    def _tokenize_word_level(self, path, max_lines=None):
        ids = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                if max_lines is not None and line_num >= max_lines:
                    break
                words = line.strip().split() + ["<eos>"]
                for word in words:
                    ids.append(self.dictionary.add_word(word))
        return ids

    # ----- subword path -----
    def _encode_file(self, path, max_lines=None):
        ids = []
        for line in _iter_lines(path, max_lines):
            ids.extend(self.tokenizer.encode(line, add_bos=False, add_eos=True))
        return ids

    @property
    def vocab_size(self):
        if self.use_subword:
            return len(self.tokenizer)
        else:
            return len(self.dictionary)


def batchify(data, batch_size, device=None, dtype=None):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    data = np.array(data, dtype=np.int64)
    nbatch = data.shape[0] // batch_size
    data = data[: nbatch * batch_size]
    # shape: (batch_size, nbatch) -> transpose -> (nbatch, batch_size)
    data = data.reshape(batch_size, nbatch).T
    return data


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    nbatch = batches.shape[0]
    seq_len = min(bptt, nbatch - 1 - i)

    data_np = batches[i : i + seq_len]
    target_np = batches[i + 1 : i + 1 + seq_len].reshape(-1)

    data = Tensor(data_np, device=device, dtype=dtype)
    target = Tensor(target_np, device=device, dtype=dtype)

    return data, target
