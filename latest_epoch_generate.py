import torch
import matplotlib.pyplot as plt

from model.transformer import PixelArtTransformer
from model.generate import generate_batch
from tokenizer.patch_tokenizer import decode, bins_to_pixels, SEQ_LEN

ckpt = torch.load("checkpoints/latest.pt", map_location="cpu")
categories = ckpt["categories"]

model = PixelArtTransformer(n_categories=len(categories))
model.load_state_dict(ckpt["model"])
model.eval()

NAMES = ["cat", "dog", "airplane", "apple", "alarm clock"]
indices = [categories.index(n) for n in NAMES]


@torch.no_grad()
def generate_with_temp(cat_idx, temperature=1.0):
    cat = torch.tensor([cat_idx], dtype=torch.long)
    tokens = torch.empty((1, 0), dtype=torch.long)

    for _ in range(SEQ_LEN):
        logits = model(cat, tokens)
        next_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    binned = decode(tokens[0].cpu().numpy())
    return bins_to_pixels(binned)


# --- Plot: greedy (temp=0) vs temp=1.0 vs temp=1.5 ---
temperatures = {"greedy": None, "temp=1.0": 1.0, "temp=1.5": 1.5}

fig, axes = plt.subplots(len(temperatures), len(NAMES),
                         figsize=(len(NAMES) * 3, len(temperatures) * 3))

for row, (label, temp) in enumerate(temperatures.items()):
    for col, (name, idx) in enumerate(zip(NAMES, indices)):
        if temp is None:
            img = generate_batch(model, [idx])[0]
        else:
            img = generate_with_temp(idx, temperature=temp)

        axes[row, col].imshow(15 - img, cmap="gray", vmin=0, vmax=15)
        axes[row, col].set_title(f"{name}\n{label}", fontsize=9)
        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
