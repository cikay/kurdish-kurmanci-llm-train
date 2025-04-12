class Lang:

    def __init__(self, text: str):
        self.stoi = {}
        self.itos = {}
        self.vocab = set(text)
        self.vocab_size = len(self.vocab)
        self.build_vocab()

    def build_vocab(self):
        self.stoi = {char: i for i, char in enumerate(self.vocab)}
        self.itos = {i: char for i, char in enumerate(self.vocab)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, indexes: list[int]) -> str:
        return "".join([self.itos[idx.item()] for idx in indexes])
