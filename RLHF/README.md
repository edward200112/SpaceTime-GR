## Rule-based PPO

Actor (Policy Model): 直接使用GRPO解读那的训练的 LLM（从 SFT 权重开始）。

Critic (Value Model): 新增组件，用于评估当前状态（Prompt + 生成的部分）的“价值”。它需要在 LLM 顶层加一个线性层（Scalar Head）。

Reference Model: 冻结的 SFT 模型，用于计算 KL 散度，防止模型忘却 SFT 学到的知识。