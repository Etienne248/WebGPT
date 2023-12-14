"""
---
title: Rotary Positional Embeddings (RoPE)
summary: >
  Annotated implementation of RoPE from paper
  RoFormer: Enhanced Transformer with Rotary Position Embedding
---

# Rotary Positional Embeddings (RoPE)

This is an implementation of
[Rotary Positional Embeddings (RoPE)](https://arxiv.org/abs/2104.09864)
in [PyTorch](https://pytorch.org).

Rotary Positional Embeddings (RoPE) encode position information of tokens
with a rotation matrix that naturally incorporates explicit relative position
dependency.

Here's [the training code](experiment.html) for training a transformer model with RoPE
 on Tiny Shakespeare dataset.
"""

import torch
from torch import nn



class RotaryPositionalEmbeddings(nn.Module):
   
    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()

        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor, start_idx = 0):
        """
        Cache $\cos$ and $\sin$ values
        """
        # Return if cache is already built
        if self.cos_cached is not None and x.shape[-2] <= self.cos_cached.shape[-2] and start_idx == 0 :
            return

        # Get sequence length
        seq_len = x.shape[-2]

        # $\Theta = {\theta_i = 10000^{-\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(start_idx,seq_len + start_idx, device=x.device).float().to(x.device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Cache them
        self.cos_cached = idx_theta2.cos()[None, None, :, :]
        self.sin_cached = idx_theta2.sin()[None, None, :, :]

    def _neg_half(self, x: torch.Tensor):
        # $\frac{d}{2}$
        d_2 = self.d // 2

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor, start_idx = 0):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x, start_idx)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        x_rope = (x_rope * self.cos_cached[:x.shape[-2]]) + (neg_half_x * self.sin_cached[:x.shape[-2]])

        return torch.cat((x_rope, x_pass), dim=-1)


def _test_rotary():

    import matplotlib.pyplot as plt
    import numpy as np
    """
    Testing RoPE with a simple example
    """
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float)
    # x= torch.full((4,10),5)
    x = x[None, None, :, :]
    print(x)

    rot_dim = 4
    rotary_pe = RotaryPositionalEmbeddings(rot_dim)
    pe = rotary_pe(x, 1)
    print(pe)

    # Plot the rotation applied to x
    d_2 = rot_dim// 2

    # Plotting the arrows
    x_values = x[0, 0, :, :d_2]
    y_values = x[0, 0, :, d_2:rot_dim]

    print(x_values)

    z = np.zeros(x_values.shape)

    plt.quiver(z,z,x_values, y_values,angles='xy', scale_units='xy', scale=1,width=0.002)

    x_values = pe[0, 0, :, :d_2]
    y_values = pe[0, 0, :, d_2:rot_dim]
    plt.quiver(z,z,x_values, y_values,angles='xy', scale_units='xy', scale=1,width=0.002,color='blue')

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Arrows for X and Y values')
    plt.grid(True)
    plt.show()
    
    


if __name__ == '__main__':
    _test_rotary()
