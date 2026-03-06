import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ----------------------------
# Device (GPU if available)
# ----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ----------------------------
# Load tokenizer
# ----------------------------

sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

text = open("data.txt", encoding="utf-8").read()

tokens = sp.encode(text)

data = torch.tensor(tokens, dtype=torch.long)

vocab_size = sp.get_piece_size()

# ----------------------------
# Model parameters
# ----------------------------

block_size = 64
embed_size = 64
heads = 4
layers = 2
batch_size = 16

# ----------------------------
# Attention Head
# ----------------------------

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        weights = q @ k.transpose(-2,-1) / (k.shape[-1]**0.5)

        weights = weights.masked_fill(self.mask[:T,:T]==0,float("-inf"))

        weights = F.softmax(weights, dim=-1)

        v = self.value(x)

        return weights @ v


# ----------------------------
# Multi-head attention
# ----------------------------

class MultiHead(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size) for _ in range(num_heads)]
        )

        self.proj = nn.Linear(head_size*num_heads, embed_size)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.heads], dim=-1)

        return self.proj(out)


# ----------------------------
# Feedforward
# ----------------------------

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size)
        )

    def forward(self,x):
        return self.net(x)


# ----------------------------
# Transformer block
# ----------------------------

class Block(nn.Module):

    def __init__(self):
        super().__init__()

        head_size = embed_size // heads

        self.attn = MultiHead(heads, head_size)
        self.ff = FeedForward()

        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self,x):

        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))

        return x


# ----------------------------
# Mini GPT Model
# ----------------------------

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.blocks = nn.Sequential(*[Block() for _ in range(layers)])

        self.ln = nn.LayerNorm(embed_size)

        self.head = nn.Linear(embed_size, vocab_size)

    def forward(self,x):

        x = self.embedding(x)

        x = self.blocks(x)

        x = self.ln(x)

        return self.head(x)


model = MiniGPT().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# ----------------------------
# Training
# ----------------------------

steps = 10000

for step in range(steps):

    ix = torch.randint(len(data)-block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    x = x.to(device)
    y = y.to(device)

    logits = model(x)

    loss = loss_fn(
        logits.view(-1,vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, loss.item())


torch.save({
    "model_state":model.state_dict(),
    "tokenizer":"tokenizer.model"
},"model.pth")

print("training complete")