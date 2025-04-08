from klpt.tokenize import Tokenize


tokenizer = Tokenize("Kurmanji", "Latin")


class Lang:
    UNK_token_i = 0
    UNK_token_s = "<unk>"

    def __init__(self, text: str):
        self.stoi = {self.UNK_token_s: self.UNK_token_i}
        self.itos = {self.UNK_token_i: self.UNK_token_s}
        self.vocab_size = 1
        self.text = text
        self.build_vocab()

    def build_vocab(self):
        all_tokens = []
        words = tokenizer.word_tokenize(self.text)
        for word in words:
            sub_words = self._get_sub_words(word)
            all_tokens.extend(sub_words)

        unique_tokens = set(all_tokens)
        # Start from 1 to reserve 0 for UNK
        for i, token in enumerate(unique_tokens, start=1):
            self.stoi[token] = i
            self.itos[i] = token

        self.vocab_size = len(self.stoi)
        print(f"Built vocabulary with {self.vocab_size} tokens")

    def encode(self, text: str) -> list[int]:
        words = tokenizer.word_tokenize(text)
        tokens = []
        for word in words:
            sub_words = self._get_sub_words(word)
            for sub_word in sub_words:
                token = self.stoi.get(sub_word, self.UNK_token_i)

                tokens.append(token)

        return tokens

    def decode(self, indexes: list[int]) -> str:
        return " ".join(
            [self.itos.get(idx.item(), self.UNK_token_s) for idx in indexes]
        )

    def _get_sub_words(self, word):
        formatted_word = word.replace("‒", "")
        sub_words = [
            sub_word for sub_word in formatted_word.split("▁") if sub_word != ""
        ]
        return sub_words
