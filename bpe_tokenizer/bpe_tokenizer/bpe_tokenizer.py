class BPETokenizer():
    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocal_from_merges()

    def _get_stats(self, ids)-> dict:
        """Get the statistics of the pairs of tokens in the text."""
        stats = {}
        for pair in zip(ids, ids[1:]):
            stats[pair] = stats.get(pair, 0) + 1
        return stats
    
    def _merge(self, ids, pair, idx):
        """Merge the pair of tokens in the text."""
        new_ids = []
        i = 0
        while i < len(ids):
            if ids[i] == pair[0] and i < len(ids) - 1 and ids[i + 1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        nb_merges = vocab_size - 256
        # text -> (unicode codes) -> bytes 
        text_bytes = text.encode("utf-8")
        # bytes -> list of integers in range 0..255
        ids = list(text_bytes)

        # we'll apply the pbe algorithm
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(nb_merges):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges # to use in encode
        self.vocab = vocab # to use in decode

    def encode(self, text):
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        while len(ids) >= 2:
            # we have now the list of the raw ids, we'll used the merges to get the pairs and reduce the seq_len
            stats = self._get_stats(ids)
            # get the pair of tokens with the lowest merge index
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        text_bytes = b"".join((self.vocab[idx] for idx in ids))
        text = text_bytes.decode("utf-8", errors="replace") 
        return text

    def _build_vocal_from_merges(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        # 1. Save the merges 
        model_file = file_prefix + ".model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("napster_tokenizer v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special_token, idx in self.special_tokens.items():
                f.write(f"{special_token} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        # 2. Save vocab
        vocab_file  = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = token.decode("utf-8", errors='replace')
                if idx in inverted_merges:
                    if idx in inverted_merges:
                        idx0, idx1 = inverted_merges[idx]
                        s0 = self.vocab[idx0].decode("utf-8", errors='replace')
                        s1 = self.vocab[idx1].decode("utf-8", errors='replace')
                        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        merges = {}
        special_tokens = {}
        idx = 256
        # we only load the merges file :
        with open(model_file, "r", encoding="utf-8") as f:
            version = f.readline().strip()
            self.pattern = f.readline().strip()
            # read special tokens :
            nb_special_tokens = int(f.readline().strip())
            for _ in range(nb_special_tokens):
                special_token, special_idx = f.readline().strip().split()
                special_tokens[special_token] = int(special_idx)
            # read merges :
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocal_from_merges()