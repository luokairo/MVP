from cgitb import text
import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        n_cond_embed=768,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        control_strength=1.0,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        # 7. visual Prompt
        # 计算每个尺度外围一圈的token数量
        outer_tokens_per_level = []
        total_outer_tokens = 0
        
        for pn in self.patch_nums:
            # 外围一圈的token数量为4*pn-4
            outer_tokens = 4*pn-4 if pn > 1 else 1
            outer_tokens_per_level.append(outer_tokens)
            total_outer_tokens += outer_tokens
        
        # 如果总token数大于20，则每个尺度只使用四个角
        use_only_corners = total_outer_tokens > 20
        self.tokens_per_scale = []
        
        token_num = 0
        self.prompt_indices = []  # 存储每个尺度应该在哪些位置添加prompt
        
        for i, pn in enumerate(self.patch_nums):
            level_indices = []
            
            if pn == 1:
                # 1x1尺度只有一个token
                level_indices = [0]
                token_num += 1
                outer_tokens = 1
            elif use_only_corners:
                # 只使用四个角：左上、右上、左下、右下
                corners = [0, pn-1, pn*(pn-1), pn*pn-1]
                level_indices = corners
                token_num += 4
                outer_tokens = 4
            else:
                # 使用外围一圈
                # 上边缘
                for j in range(pn):
                    level_indices.append(j)
                # 右边缘
                for j in range(1, pn-1):
                    level_indices.append(j*pn + (pn-1))
                # 下边缘
                for j in range(pn-1, -1, -1):
                    level_indices.append((pn-1)*pn + j)
                # 左边缘
                for j in range(pn-2, 0, -1):
                    level_indices.append(j*pn)
                
                token_num += len(level_indices)
                outer_tokens = len(level_indices)
            
            self.prompt_indices.append(level_indices)
            self.tokens_per_scale.append(outer_tokens)
            
        print(f"Using {'corner' if use_only_corners else 'outer border'} tokens, total: {token_num}")
        total_prompt_tokens = token_num
        self.visual_prompt = nn.Parameter(torch.empty(total_prompt_tokens, self.C))
        nn.init.trunc_normal_(self.visual_prompt.data, mean=0, std=init_std)
        
        # 添加prompt控制层和控制强度
        self.layer_internal = depth // 3  # 将模型分为3个阶段，每个阶段均匀添加prompt
        self.control_strength = control_strength
        self.prompt_layers = nn.ModuleList()
        for _ in range(3):
            self.prompt_layers.append(nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(self.C, self.C, bias=False)
            ))
            
        # 初始化prompt控制层参数
        for prompt_layer in self.prompt_layers:
            nn.init.zeros_(prompt_layer[-1].weight)  # 将最后的线性层权重初始化为0
        # 8. text embedding
        print("--------------------------------")
        print(type(n_cond_embed))
        print(n_cond_embed)
        print("--------------------------------")
        self.noise = nn.Embedding(1, n_cond_embed)
        nn.init.trunc_normal_(self.noise.weight.data, mean=0, std=init_std)
        self.cond_proj = nn.Linear(n_cond_embed, self.C) 

    def create_feature_maps_with_visual_prompt(self, patch_nums, visual_prompt, batch_size=None):
        """
        创建包含visual prompt的特征图，并将其转换为token序列
        
        Args:
            patch_nums: 各尺度大小列表
            visual_prompt: 可学习的prompt参数
            batch_size: 批次大小，如果提供则扩展为(B,L,C)形式
            
        Returns:
            如果batch_size为None: (L,C)形状的token序列
            如果batch_size不为None: (B,L,C)形状的token序列
        """
        feature_maps = []
        token_idx = 0  # 跟踪当前使用的token索引
        
        for i, pn in enumerate(patch_nums):
            # 创建全0特征图，现在有通道维度
            feature_map = torch.zeros((pn, pn, self.C), device=visual_prompt.device)
            
            # 对于1x1的特殊情况
            if pn == 1:
                feature_map[0, 0, :] = visual_prompt[token_idx]
                token_idx += 1
                feature_maps.append(feature_map.reshape(-1, self.C))  # 将(1,1,C)转换为(1,C)
                continue
            
            # 获取该尺度可用的token数量
            available_tokens = self.tokens_per_scale[i]
            
            # 计算特征图外围的总位置数
            perimeter = 4*pn-4
            
            if available_tokens >= perimeter:
                # 如果token足够填满整个外围，按顺时针方式填充
                coords = []
                # 上边缘
                for j in range(pn):
                    coords.append((0, j))
                # 右边缘（不包括右上角）
                for j in range(1, pn):
                    coords.append((j, pn-1))
                # 下边缘（从右到左，不包括右下角）
                for j in range(pn-2, -1, -1):
                    coords.append((pn-1, j))
                # 左边缘（从下到上，不包括左下角和左上角）
                for j in range(pn-2, 0, -1):
                    coords.append((j, 0))
                    
                for j in range(len(coords)):
                    row, col = coords[j]
                    feature_map[row, col, :] = visual_prompt[token_idx]
                    token_idx += 1
            else:
                # 如果token不足以填满整个外围，集中在四个角上并向外延伸
                tokens_per_corner = available_tokens // 4
                remainder = available_tokens % 4
                
                # 计算每个角落延伸的长度
                corner_size = max(1, tokens_per_corner // 2 + 1)  # 至少延伸1个位置
                
                # 左上角
                corner_tokens = tokens_per_corner + (1 if remainder > 0 else 0)
                tokens_used = 0
                
                for k in range(min(corner_size, pn)):
                    if tokens_used < corner_tokens:
                        feature_map[0, k, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
                
                for k in range(1, min(corner_size, pn)):
                    if tokens_used < corner_tokens:
                        feature_map[k, 0, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
                
                # 右上角
                corner_tokens = tokens_per_corner + (1 if remainder > 1 else 0)
                tokens_used = 0
                
                for k in range(pn-1, max(pn-1-corner_size, -1), -1):
                    if tokens_used < corner_tokens:
                        feature_map[0, k, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
                
                for k in range(1, min(corner_size, pn)):
                    if tokens_used < corner_tokens:
                        feature_map[k, pn-1, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
                
                # 左下角
                corner_tokens = tokens_per_corner + (1 if remainder > 2 else 0)
                tokens_used = 0
                
                for k in range(min(corner_size, pn)):
                    if tokens_used < corner_tokens:
                        feature_map[pn-1, k, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
                
                for k in range(pn-2, max(pn-1-corner_size, -1), -1):
                    if tokens_used < corner_tokens:
                        feature_map[k, 0, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
                
                # 右下角
                corner_tokens = tokens_per_corner
                tokens_used = 0
                
                for k in range(pn-1, max(pn-1-corner_size, -1), -1):
                    if tokens_used < corner_tokens:
                        feature_map[pn-1, k, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
                
                for k in range(pn-2, max(pn-1-corner_size, -1), -1):
                    if tokens_used < corner_tokens:
                        feature_map[k, pn-1, :] = visual_prompt[token_idx]
                        token_idx += 1
                        tokens_used += 1
            
            # 将特征图转换为token序列形式 (ph*pw, C)
            feature_maps.append(feature_map.reshape(-1, self.C))
        
        # 将所有尺度的token序列拼接为一个完整序列 (L, C)
        token_sequence = torch.cat(feature_maps, dim=0)
        
        # 如果提供了batch_size，则扩展为多批次形式 (B, L, C)
        if batch_size is not None:
            token_sequence = token_sequence.unsqueeze(0).expand(batch_size, -1, -1)
            
        return token_sequence
        
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        text_embedding: Optional[torch.Tensor] = None,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, control_strength=None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :param control_strength: prompt控制强度，如果为None则使用默认值
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        # 设置控制强度
        actual_control_strength = control_strength if control_strength is not None else self.control_strength
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        # for text-to-image generation, we need to generate a visual prompt
        if text_embedding is not None:
            noise = self.noise(torch.tensor(0, device=text_embedding.device)).unsqueeze(0).expand(B, -1).to(text_embedding.dtype)
            sos = cond_BD = self.cond_proj(torch.cat((text_embedding, noise), dim=0))
        
        # 生成完整的visual prompt token序列 (L,C)
        visual_prompt_tokens = self.create_feature_maps_with_visual_prompt(
            self.patch_nums, self.visual_prompt
        )
        
        # 处理visual prompt用于各层之间
        prompt_features = []
        for i in range(3):
            prompt_features.append(self.prompt_layers[i](visual_prompt_tokens))
        
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        # 添加第一个尺度的visual prompt
        next_token_map = next_token_map + visual_prompt_tokens[:self.first_l].unsqueeze(0).expand(2 * B, -1, -1)
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            # 计算每个transformer block并在合适的地方添加prompt
            for i, b in enumerate(self.blocks):
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                # 在每个阶段的适当位置添加visual prompt
                if i % self.layer_internal == self.layer_internal - 1 and i < self.depth - 1:
                    stage_idx = i // self.layer_internal
                    if stage_idx < 3:  # 确保不超出范围
                        # 为两个batch (原始和CFG) 都添加prompt
                        prompt_to_add = prompt_features[stage_idx][:cur_L].unsqueeze(0).expand(2 * B, -1, -1)
                        x = x + actual_control_strength * prompt_to_add
                        
            logits_BlV = self.get_logits(x, cond_BD)
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
                
                # 添加下一个尺度的visual prompt
                next_token_map = next_token_map + visual_prompt_tokens[cur_L:cur_L + self.patch_nums[si+1]**2].unsqueeze(0).expand(2 * B, -1, -1)
        
        for b in self.blocks: b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor, text_embedding: torch.Tensor=None) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]

        # 生成visual prompt token序列 (B,L,C)
        visual_prompt_tokens = self.create_feature_maps_with_visual_prompt(
            self.patch_nums, self.visual_prompt, batch_size=B
        )
        
        # 处理visual prompt用于各层之间
        prompt_features = []
        for i in range(3):
            prompt_features.append(self.prompt_layers[i](visual_prompt_tokens))

        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            if text_embedding is not None:
                indexs = torch.randperm(B)[:int(B*0.1)]
                noise = self.noise(torch.tensor(0, device=label_B.device)).unsqueeze(0).to(text_embedding.dtype)
                text_embedding[indexs,:] = noise
                text_embedding = text_embedding.to(self.cond_proj.weight.dtype)
                text_embedding = self.cond_proj(text_embedding)
                cond_BD = text_embedding

                sos = text_embedding.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)   # (B, 1, C)
                
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
            
            # 添加第一级visual prompt到输入序列
            x_BLC = x_BLC + visual_prompt_tokens[:, :ed, :]
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        # 转换prompt特征到正确的数据类型
        for i in range(3):
            prompt_features[i] = prompt_features[i].to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            
            # 在每个阶段的适当位置添加visual prompt
            if i % self.layer_internal == self.layer_internal - 1 and i < self.depth - 1:
                stage_idx = i // self.layer_internal
                if stage_idx < 3:  # 确保不超出范围
                    x_BLC = x_BLC + self.control_strength * prompt_features[stage_idx][:, :ed, :]
                    
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        for si, pn in enumerate(self.patch_nums):
            logits_BlV = x_BLC[:, cur_L:cur_L+pn*pn, :]
            idx_Bl = torch.argmax(logits_BlV, dim=-1)
            h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, _ = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw
            )
            if si == len(self.patch_nums) // 2:
                f_hat_half = f_hat

            cur_L += pn*pn
        
        img_half = self.vae_proxy[0].fhat_to_img(f_hat_half).add_(1).mul_(0.5)
        img = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)
        return x_BLC, img_half, img    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
        control_strength=1.0,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            control_strength=control_strength,
        )
