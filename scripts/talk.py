"""Talk to the 5M-parameter Saryu LM (v4b checkpoint, enwik8, bpc ~1.78).

Character-level completion, not chat: give it a prompt, it continues in Wikipedia dialect.
Usage:
    python scripts/talk.py                          # canned prompts, one sample each
    python scripts/talk.py "The history of India"   # your prompt
    python scripts/talk.py -i                       # interactive loop (Ctrl+C to exit)
Sampling: temperature + top-k over characters. The recurrent state is O(1), so generation
cost per character is flat regardless of how long the conversation gets.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(4)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import saryu.model as _m  # noqa: E402

CORPUS = os.path.join(ROOT, 'corpus', 'enwik8')


VOCAB = os.path.join(ROOT, 'saryu', 'enwik8_vocab.json')


def build_vocab(vocab_min=100):
    if os.path.exists(VOCAB):              # the same 480 code points, saved so the corpus is optional
        keep = json.load(open(VOCAB))['codepoints']
    else:
        raw = open(CORPUS, 'rb').read().decode('utf-8', errors='ignore')
        cps = np.frombuffer(raw.encode('utf-32-le'), dtype=np.uint32)
        uniq, counts = np.unique(cps, return_counts=True)
        keep = np.sort(uniq[counts >= vocab_min])
    stoi = {int(c): i + 1 for i, c in enumerate(keep)}
    itos = {i + 1: chr(int(c)) for i, c in enumerate(keep)}
    itos[0] = '?'
    return stoi, itos, len(keep) + 1


print('loading vocab + checkpoint...', flush=True)
stoi, itos, V = build_vocab()
ck = torch.load(os.path.join(ROOT, 'checkpoints', 'saryu_v4b_last.pt'), map_location='cpu', weights_only=False)
model = _m.SaryuV3LM(V, ck['d'], 4, use_vemb=True)
model.load_state_dict(ck['state'])
model.eval()
n = sum(p.numel() for p in model.parameters())
print('loaded arm {!r}: {:,} params, vocab {}'.format(ck['arm'], n, V), flush=True)


def encode(text):
    return torch.tensor([[stoi.get(ord(c), 0) for c in text]])


@torch.no_grad()
def generate(prompt, n_chars=400, temp=0.8, topk=40, ctx=512):
    idx = encode(prompt)
    out = []
    for _ in range(n_chars):
        window = idx[:, -ctx:]
        # pad so chunkwise sees a multiple of CHUNK
        C = _m.CHUNK
        pad = (-window.shape[1]) % C
        if pad:
            window = torch.cat([torch.zeros(1, pad, dtype=torch.long), window], 1)
        logits = model(window)[0, -1] / temp
        v, _ix = torch.topk(logits, topk)
        logits[logits < v[-1]] = -1e30
        p = torch.softmax(logits, -1)
        nxt = int(torch.multinomial(p, 1))
        out.append(itos.get(nxt, '?'))
        idx = torch.cat([idx, torch.tensor([[nxt]])], 1)
    return ''.join(out)


if __name__ == '__main__':
    args = sys.argv[1:]
    if args and args[0] == '-i':
        print('interactive: type a prompt, empty line to quit')
        while True:
            try:
                prompt = input('\n>>> ')
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt.strip():
                break
            print(prompt + generate(prompt), flush=True)
    else:
        prompts = args if args else [
            'The history of India begins with ',
            'In mathematics, a group is ',
            '== Agriculture ==\nFarming in ',
        ]
        for pr in prompts:
            print('\n' + '=' * 70)
            print('PROMPT: {!r}'.format(pr))
            print('-' * 70)
            print(pr + generate(pr), flush=True)
