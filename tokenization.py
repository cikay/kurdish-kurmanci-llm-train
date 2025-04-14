class Lang:

    def __init__(self, vocab: set | None = None):
        if vocab is None:
            vocab = set()

        self.stoi = {}
        self.itos = {}
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.build_vocab()

    def build_vocab(self):
        self.stoi = {char: i for i, char in enumerate(self.vocab)}
        self.itos = {i: char for i, char in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, indexes: list[int]) -> str:
        return "".join([self.itos[idx.item()] for idx in indexes])

    def state_dict(self):
        return {
            "stoi": self.stoi,
            "itos": self.itos,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        instance = cls()
        instance.stoi = state_dict["stoi"]
        instance.itos = state_dict["itos"]
        instance.vocab = state_dict["vocab"]
        instance.vocab_size = state_dict["vocab_size"]
        return instance
