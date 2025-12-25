移除：所有关于 hierarchy_mask、raw_target_code、level2_token_ids 的逻辑（因为我们不再用 SID）。

保留：CoIN 逻辑（正样本、增强样本、负样本），这是利用 SASRec 知识的关键。

Tokenizer：改回加载基础模型（如 Qwen/Llama），不再加载 Aligned Checkpoint。


使用困难负样本，让SASRec作为Teacher Model