import torch
import torch.nn.functional as F


def clip_contrastive_loss(text_embedding, img_embedding, temperature=0.07):
    """
    基础CLIP对比损失，通过最大化正样本对的相似度，最小化负样本对的相似度来实现文本和图像的对齐
    
    参数:
        text_embedding: 文本嵌入 [batch_size, embed_dim]
        img_embedding: 图像嵌入 [batch_size, embed_dim]
        temperature: 温度参数，控制softmax的平滑度
    """
    # 归一化嵌入
    text_embedding = F.normalize(text_embedding, dim=-1)
    img_embedding = F.normalize(img_embedding, dim=-1)
    
    # 计算余弦相似度矩阵 [batch_size, batch_size]
    logits = torch.matmul(text_embedding, img_embedding.t()) / temperature
    
    # 对角线上是正样本对
    batch_size = text_embedding.shape[0]
    labels = torch.arange(batch_size, device=text_embedding.device)
    
    # 计算文本到图像和图像到文本的交叉熵损失
    loss_t2i = F.cross_entropy(logits, labels)
    loss_i2t = F.cross_entropy(logits.t(), labels)
    
    # 综合损失
    loss = (loss_t2i + loss_i2t) / 2.0
    
    return loss

def enhanced_clip_alignment_loss(text_embedding, img_embedding, text_half_embedding=None, img_half_embedding=None, temperature=0.07, distillation_weight=0.3):
    """
    增强型CLIP对齐损失，结合了对比损失、硬负样本挖掘和知识蒸馏
    
    参数:
        text_embedding: 文本嵌入 [batch_size, embed_dim]
        img_embedding: 图像嵌入 [batch_size, embed_dim]
        text_half_embedding: 可选的第二个文本嵌入用于知识蒸馏
        img_half_embedding: 可选的第二个图像嵌入用于知识蒸馏
        temperature: 温度参数，控制softmax的平滑度
        distillation_weight: 知识蒸馏损失的权重
    """
    # 归一化嵌入
    text_embedding = F.normalize(text_embedding, dim=-1)
    img_embedding = F.normalize(img_embedding, dim=-1)
    
    batch_size = text_embedding.shape[0]
    device = text_embedding.device
    
    # 1. 基础对比损失
    similarity = torch.matmul(text_embedding, img_embedding.t()) / temperature
    labels = torch.arange(batch_size, device=device)
    
    # 对角线掩码，用于识别正样本对
    pos_mask = torch.eye(batch_size, device=device).bool()
    
    # 计算交叉熵损失
    loss_t2i = F.cross_entropy(similarity, labels)
    loss_i2t = F.cross_entropy(similarity.t(), labels)
    contrastive_loss = (loss_t2i + loss_i2t) / 2.0
    
    # 2. 硬负样本挖掘 - 找出最容易混淆的负样本
    with torch.no_grad():
        # 获取每个样本最相似的负样本的索引
        neg_mask = ~pos_mask
        text_to_img_sim = similarity.clone()
        text_to_img_sim[pos_mask] = -float('inf')  # 排除正样本
        hardest_img_idx = torch.argmax(text_to_img_sim, dim=1)
        
        img_to_text_sim = similarity.t().clone()
        img_to_text_sim[pos_mask] = -float('inf')  # 排除正样本
        hardest_text_idx = torch.argmax(img_to_text_sim, dim=1)
    
    # 针对硬负样本的额外对比损失
    hard_neg_loss = 0.0
    for i in range(batch_size):
        # 降低当前文本与最混淆图像的相似度
        hard_neg_loss += F.softplus(similarity[i, hardest_img_idx[i]] - similarity[i, i])
        # 降低当前图像与最混淆文本的相似度
        hard_neg_loss += F.softplus(similarity[hardest_text_idx[i], i] - similarity[i, i])
    hard_neg_loss = hard_neg_loss / (2 * batch_size)
    
    # 3. 知识蒸馏（如果提供了half embeddings）
    distillation_loss = 0.0
    if text_half_embedding is not None and img_half_embedding is not None:
        text_half_embedding = F.normalize(text_half_embedding, dim=-1)
        img_half_embedding = F.normalize(img_half_embedding, dim=-1)
        
        # 文本到文本的一致性
        text_text_sim = F.cosine_similarity(text_embedding, text_half_embedding, dim=1).mean()
        # 图像到图像的一致性
        img_img_sim = F.cosine_similarity(img_embedding, img_half_embedding, dim=1).mean()
        
        # 增加文本与图像表示的一致性
        distillation_loss = 2.0 - text_text_sim - img_img_sim
    
    # 综合所有损失
    total_loss = contrastive_loss + 0.5 * hard_neg_loss + distillation_weight * distillation_loss
    
    return total_loss, {
        'contrastive_loss': contrastive_loss.item(),
        'hard_neg_loss': hard_neg_loss.item(),
        'distillation_loss': distillation_loss.item() if isinstance(distillation_loss, torch.Tensor) else distillation_loss
    }