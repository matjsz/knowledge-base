---
title: "Attention Is All You Need: A Re-implementation"
date: "2026-02-01"
type: "research"  # Options: research, engineering, essay, project
tags: ["NLP", "Transformer", "PyTorch", "Math"]
summary: "Dissecting the mathematical foundations of the Transformer architecture and implementing the multi-head attention mechanism from scratch."
reading_time: "15 min"
citations: true   # Toggles the academic citation footer in UI
math: true        # Tells the frontend to load KaTeX/MathJax
---

# Introduction
The core of the Transformer is the Scaled Dot-Product Attention. 

## The Mathematics
Mathematically, we define the attention function as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where $d_k$ is the dimension of the keys. We divide by $\sqrt{d_k}$ to prevent the dot products from growing too large in magnitude.

## Implementation
Here is how we translate that equation into PyTorch:

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    p_attn = scores.softmax(dim=-1)
    return torch.matmul(p_attn, value), p_attn
```