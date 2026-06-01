---
layout: post
title:  "You Can Learn Tokenization End-to-End with Reinforcement Learning"
---

My work on learning to draw token boundaries using time-discounted score function estimates has been accepted into ICML 2026! [[paper](https://openreview.net/forum?id=0rXBwsTWDB)], [[code](https://github.com/SamD770/bitter-lesson-tokenization)], [[Spotlight talk at ICLR 2026 workshop](https://slideslive.com/39064330)]

## Adding Token-Strided Convolutions to NanoGPT

Unrelated to the paper, I tried to add byte-level information to standard tokenized LLMs using token-strided convolutions on byte embeddings. This roughly corresponds to applying a byte projection + linear projection to the following:

|   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|
|   |   |   |   | o |   |   |
|   |   |   |   | l |   |   |
|   |   |   |   | u |   |   |
| t |   |   | _ | t |   | _ |
| o | _ | i | c | i | _ | c |
| k | s | d | o | o | a | o |
| e | t | e | n | n | r | o |
| n | r | d | v | s | e | l |

An interesting negative result is that this doesn't [meaningfully affect the NanoGPT speedrun baseline in terms of downstream loss](https://github.com/SamD770/modded-nanogpt-private).
