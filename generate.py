import torch
import torch.nn as nn
import torch.nn.functional as F

checkpoint = torch.load("model.pth")

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

vocab_size = len(stoi)

block_size = 256
embed_size = 256


class SelfAttention(nn.Module):

    def __init__(self, embed_size, block_size):
        super().__init__()

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        T = x.shape[0]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        weights = Q @ K.transpose(-2, -1) / (embed_size ** 0.5)

        weights = weights.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(weights, dim=-1)

        return weights @ V


class FeedForward(nn.Module):

    def __init__(self, embed_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )

    def forward(self, x):
        return self.net(x)


class TinyGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.ln1 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, block_size)

        self.ln2 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)

        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return self.linear(x)


model = TinyGPT()
model.load_state_dict(checkpoint["model_state"])
model.eval()


def encode(s):
    return [stoi[c] for c in s]


def decode(l):
    return ''.join([itos[i] for i in l])


prompt = input("Enter prompt: ")

idx = torch.tensor(encode(prompt), dtype=torch.long)

temperature = 0.7
top_k = 40


for _ in range(400):

    idx_cond = idx[-block_size:]

    logits = model(idx_cond)

    logits = logits[-1] / temperature

    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, float("-inf"))
        logits_filtered[indices] = logits[indices]
        logits = logits_filtered

    probs = torch.softmax(logits, dim=0)

    next_id = torch.multinomial(probs, 1).item()

    idx = torch.cat([idx, torch.tensor([next_id])])


print("\nGenerated text:\n")
print(decode(idx.tolist()))