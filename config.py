# model
vocab_size: int = 6400
max_batch_size: int = 32
max_seq_len: int = 512
dim: int = 512
# dim: int = 640
n_head: int = 8
n_layer: int = 8
eps: float = 1e-6
use_moe: bool = False

# mla
q_lora_rank: int = 0
kv_lora_rank: int = 128
qk_nope_head_dim: int = 32
qk_rope_head_dim: int = 16
v_head_dim: int = 32
# q_lora_rank: int = 0
# kv_lora_rank: int = 160
# qk_nope_head_dim: int = 40
# qk_rope_head_dim: int = 20
# v_head_dim: int = 40

# mlp
inter_dim = 1408
# inter_dim = 1760

# moe
n_route_expert: int = 4
n_share_expert: int = 1
n_activate_expert: int = 2
score_func: str = 'softmax'
aux_alpha: float = 0.1
route_scale: float = 1.0
moe_inter_dim: int = 744
# moe_inter_dim: int = 930

# rope
beta_fast: int = 32
beta_slow: int = 1
rope_theta: float = 10000.0
rope_factor: float = 40

# train
epoch = 3
num_workers = 8
lr = 5e-4
accumulation_steps = 8
grad_clip = 1.0
print_step = 100
save_step = 100

# inference
max_new_tokens = max_seq_len * 8
temperature = 0.85
top_p = 0.85
rp = 1.0
use_cache = True
history_cnt = 0