import math

import torch
import torch.nn as nn
from einops import rearrange


def precompute_rope(config):
    dim = config.qk_rope_head_dim
    seq_len = config.max_seq_len * 8
    beta_fast = config.beta_fast
    beta_slow = config.beta_slow
    theta = config.rope_theta
    factor = config.rope_factor

    def find_correction_dim(num_rotations, dim, theta, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(theta))

    def find_correction_range(beta_fast, beta_slow, dim, theta, max_seq_len):
        low = math.floor(find_correction_dim(beta_fast, dim, theta, max_seq_len))
        high = math.ceil(find_correction_dim(beta_slow, dim, theta, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(low, high, dim):
        if low == high:
            high += 0.001
        linear_func = (torch.arange(dim, dtype = torch.float32) - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype = torch.float32) / dim))
    if seq_len > config.max_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, theta, config.max_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rope_emb(x: torch.Tensor, rope_emb: torch.Tensor):
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    rope_emb = rope_emb.view(1, rope_emb.shape[0], 1, rope_emb.shape[1])
    y = torch.view_as_real(x * rope_emb).flatten(3)
    return y.to(dtype)


class RMSNorm(nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6
                 ):
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self,
                x: torch.Tensor
                ):
        _norm = x * torch.rsqrt(x.float().pow(2).mean(dim = -1, keepdim = True) + self.eps)
        out = _norm.type_as(x) * self.weight
        return out


class MLA(nn.Module):

    def __init__(self,
                 config
                 ):
        super().__init__()

        self.dim = config.dim
        self.n_head = config.n_head
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim

        if self.q_lora_rank > 0:
            self.to_q = nn.Sequential(
                nn.Linear(self.dim, self.q_lora_rank, bias = False),
                RMSNorm(self.q_lora_rank),
                nn.Linear(self.q_lora_rank, self.n_head * self.qk_head_dim, bias = False)
            )
        else:
            self.to_q = nn.Linear(self.dim, self.n_head * self.qk_head_dim, bias = False)
        self.to_kv_a = nn.Linear(self.dim, self.qk_rope_head_dim + self.kv_lora_rank, bias = False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.to_kv_b = nn.Linear(self.kv_lora_rank, self.n_head * (self.qk_nope_head_dim + self.v_head_dim),
                                 bias = False)
        self.scale = self.qk_head_dim ** -0.5
        self.to_out = nn.Linear(self.n_head * self.v_head_dim, self.dim, bias = False)

        self.register_buffer('k_cache',
                             torch.zeros(config.max_batch_size, config.max_seq_len * 8, self.n_head, self.qk_head_dim),
                             persistent = False
                             )
        self.register_buffer('v_cache',
                             torch.zeros(config.max_batch_size, config.max_seq_len * 8, self.n_head, self.v_head_dim),
                             persistent = False
                             )
        self.register_buffer('attn_mask',
                             torch.triu(torch.full((config.max_seq_len, config.max_seq_len), -1e8), 1),
                             persistent = False
                             )

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                rope_emb: torch.Tensor,
                use_cache: bool = False
                ):
        # x:[b, n, d]
        # rope_emb:[n, rope_d // 2]
        b, n, _ = x.shape
        # q
        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b n h d', h = self.n_head)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim = -1)
        q_rope = apply_rope_emb(q_rope, rope_emb)

        # k
        kv = self.to_kv_a(x)
        k_rope, kv = kv.split([self.qk_rope_head_dim, self.kv_lora_rank], dim = -1)
        k_rope = apply_rope_emb(k_rope.unsqueeze(dim = 2), rope_emb)

        # attn
        q = torch.cat((q_nope, q_rope), dim = -1)
        kv = self.to_kv_b(self.kv_norm(kv))
        kv = rearrange(kv, 'b n (h d) -> b n h d', h = self.n_head)
        k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim = -1)
        k = torch.cat((k_nope, k_rope.expand(-1, -1, self.n_head, -1)), dim = -1)
        if use_cache:
            self.k_cache[:b, start_pos: start_pos + n] = k
            self.v_cache[:b, start_pos: start_pos + n] = v
            q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d'),
                          (q, self.k_cache[:b, :start_pos + n], self.v_cache[:b, :start_pos + n]))
        else:
            q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d'), (q, k, v))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if n > 1:
            attn += self.attn_mask[:n, :n]
        attn = torch.softmax(attn, dim = -1)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class MLP(nn.Module):

    def __init__(self,
                 dim,
                 inter_dim,
                 use_bias = False
                 ):
        super().__init__()

        self.dim = dim
        self.inter_dim = inter_dim
        self.linear1 = nn.Linear(self.dim, self.inter_dim, bias = use_bias)
        self.linear2 = nn.Linear(self.dim, self.inter_dim, bias = use_bias)
        self.linear3 = nn.Linear(self.inter_dim, self.dim, bias = use_bias)
        self.silu = nn.SiLU()

    def forward(self,
                x: torch.Tensor
                ):
        out = self.linear3(self.silu(self.linear1(x)) * self.linear2(x))
        return out


class Gate(nn.Module):

    def __init__(self,
                 config
                 ):
        super().__init__()

        self.n_activate_expert = config.n_activate_expert
        self.score_func = config.score_func
        self.aux_alpha = config.aux_alpha
        self.route_scale = config.route_scale

        self.linear = nn.Linear(config.dim, config.n_route_expert, bias = False)
        self.bias = nn.Parameter(torch.empty(config.n_route_expert))

    def forward(self,
                x: torch.Tensor
                ):
        score = self.linear(x)
        if self.score_func == 'softmax':
            score = torch.softmax(score, dim = -1)
        else:
            score = torch.sigmoid(score)
        ori_score = score
        score = score + self.bias # Auxiliary-Loss-Free Load Balancing

        idx = torch.topk(score, k = self.n_activate_expert, dim = -1)[1]
        weight = ori_score.gather(dim = 1, index = idx)

        # Complementary Sequence-Wise Auxiliary Loss
        if self.training and self.aux_alpha > 0.0:
            p = ori_score.mean(dim = 0)
            expert_mask = torch.zeros(*ori_score.shape, device = ori_score.device)
            expert_mask.scatter_(1, idx, 1.0)
            f = expert_mask.sum(dim = 0) / ori_score.shape[0]
            aux_loss = (p * f).sum() * self.aux_alpha
        else:
            aux_loss = 0.0

        weight *= self.route_scale
        return weight, idx, aux_loss


class MOE(nn.Module):

    def __init__(self,
                 config
                 ):
        super().__init__()

        self.n_route_expert = config.n_route_expert

        self.gate = Gate(config)
        self.expert = nn.ModuleList([
            MLP(config.dim, config.inter_dim, use_bias = True) for _ in range(self.n_route_expert)
        ])
        self.share_expert = MLP(config.dim, config.n_share_expert * config.moe_inter_dim)

    def forward(self,
                x: torch.Tensor,
                ):
        shape = x.shape
        x = x.view(-1, shape[-1])
        weight, idx, aux_loss = self.gate(x)

        counts = torch.bincount(idx.flatten(), minlength=self.n_route_expert).tolist()
        y = torch.zeros_like(x)
        for i, expert in enumerate(self.expert):
            if counts[i] == 0:
                continue
            row, col = torch.where(idx == i)
            y[row] += expert(x[row]) * weight[row, col, None]
        z = self.share_expert(x)
        out = (y + z).view(shape)
        return out, aux_loss


class Block(nn.Module):

    def __init__(self,
                 config
                 ):
        super().__init__()

        self.use_moe = config.use_moe

        self.attn = MLA(config)
        self.ffn = MOE(config) if self.use_moe else MLP(config.dim, config.inter_dim)
        self.attn_norm = RMSNorm(config.dim)
        self.ffn_norm = RMSNorm(config.dim)

    def forward(self,
                x: torch.Tensor,
                start_pos: int,
                rope_emb: torch.Tensor,
                use_cache: bool = False
                ):
        x = x + self.attn(self.attn_norm(x), start_pos, rope_emb, use_cache)
        if self.use_moe:
            x, aux_loss = self.ffn(self.ffn_norm(x))
            return x, aux_loss
        else:
            x = x + self.ffn(self.ffn_norm(x))
            return x, 0.0


class Transformer(nn.Module):

    def __init__(self,
                 config
                 ):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.dim = config.dim

        self.token_embedding = nn.Embedding(self.vocab_size, self.dim)
        self.layers = nn.ModuleList([
            Block(config) for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(self.dim)
        self.to_out = nn.Linear(self.dim, self.vocab_size, bias = False)

        self.register_buffer('rope_embedding',
                             precompute_rope(config),
                             persistent = False,
                             )

    def forward(self,
                x: torch.Tensor,
                start_pos: int = 0,
                use_cache: bool = False
                ):
        # x:[b, n]
        _, n = x.shape
        h = self.token_embedding(x)
        rop_emb = self.rope_embedding[start_pos: start_pos + n]
        aux_loss_list = []
        for layer in self.layers:
            h, aux_loss = layer(h, start_pos, rop_emb, use_cache)
            aux_loss_list.append(aux_loss)
        out = self.to_out(self.norm(h))
        return out, sum(aux_loss_list)

    @torch.inference_mode()
    def inference(self,
                  x: torch.Tensor,
                  eos_token_id: int,
                  max_new_tokens: int,
                  temperature: float,
                  top_p: float,
                  rp: float,
                  use_cache: bool = True
                  ):
        n, first = x.shape[1], True
        while x.shape[1] < max_new_tokens - 1:
            if first:
                out, _ = self.forward(x, start_pos = 0, use_cache = use_cache)
            else:
                out, _ = self.forward(x[:, -1:], start_pos = x.shape[1] - 1, use_cache = use_cache)

            logits = out[:, -1, :]
            logits[:, list(set(x.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sort_logits, sort_idx = torch.sort(logits, descending = True)
                sort_prob = torch.softmax(sort_logits, dim = -1)
                cum_prob = torch.cumsum(sort_prob, dim = -1)
                cut_idx = torch.argmax((cum_prob >= top_p).float())
                keep_mask = torch.zeros_like(logits)
                keep_mask[:, sort_idx[:, :cut_idx + 1]] = 1
                logits = logits.masked_fill(keep_mask == 0, -float('Inf'))
            x_next = torch.multinomial(torch.softmax(logits, dim = -1), num_samples = 1)
            x = torch.cat((x, x_next), dim = 1)
            yield x[:, n:]
            if x_next.item() == eos_token_id:
                break