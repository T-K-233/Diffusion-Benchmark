from typing import Union, Optional, Tuple
import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerForDiffusion(nn.Module):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = 4,
            cond_dim: int = 10,
            n_layer: int = 6,
            n_head: int = 8,
            n_emb: int = 256,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            device: str = "cuda"
        ) -> None:
        super().__init__()

        # constants
        self.T_cond = 1 + n_obs_steps
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond

        self.device = device


        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb, device=device)
        self.pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb, device=device))

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb, device=device)
        
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, self.T_cond, n_emb, device=device))
        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb, device=device),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb, device=device),
        )
        # decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4*n_emb,
            activation="gelu",
            batch_first=True,
            norm_first=True, # important for stability
            device=device
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=n_layer
        )

        # attention mask
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
        # therefore, the upper triangle should be -inf and others (including diag) should be 0.
        sz = horizon
        
        # boolean mask
        # mask = nn.Transformer.generate_square_subsequent_mask(sz, device=self.device)

        # float mask
        mask = ~nn.Transformer.generate_square_subsequent_mask(sz, dtype=torch.bool, device=self.device)


        self.register_buffer("mask", mask)
        
        t, s = torch.meshgrid(
            torch.arange(sz, device=device),
            torch.arange(self.T_cond, device=device),
            indexing="ij"
        )

        # boolean mask
        mask = t >= (s-1) # add one dimension since time is the first token in cond

        # float mask
        #mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        
        self.register_buffer("memory_mask", mask)

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb, device=device)
        self.head = nn.Linear(n_emb, output_dim, device=device)
        
        # init
        print("number of parameters: %e" % sum(p.numel() for p in self.parameters()))


    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (4, 8, 16)
        timestep: (1,)
        cond: (4, 4, 10)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        # (4, 1, 256)
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        # process input
        # (4, 8, 256)
        input_emb = self.input_emb(sample)
        # (4, 4, 256)
        cond_obs_emb = self.cond_obs_emb(cond)
        
        
        # encoder
        # (4, 5, 256)
        cond_embeddings = torch.cat([time_emb, cond_obs_emb], dim=1)
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]  # each position maps to a (learnable) vector
        x = cond_embeddings + position_embeddings
        # (4. 5. 256)
        x = self.encoder(x)
        memory = x
        
        # decoder
        token_embeddings = input_emb
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = token_embeddings + position_embeddings
        # (4, 8, 256)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_mask=self.memory_mask
        )
        
        # head

        # (4, 8, 256)
        x = self.ln_f(x)
        
        # (4, 8, 16)
        x = self.head(x)
        return x
