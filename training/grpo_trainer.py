"""
GRPO (Group Relative Policy Optimization) Trainer for HierGR-SeqRec

基于 MiniOneRec 实现，简化并适配到 HierGR-SeqRec 项目
"""

import os
import torch
import torch.nn as nn
import math
from typing import Optional, Union, Callable, List, Dict, Any
from collections import defaultdict
from torch.utils.data import Sampler, Dataset
from tqdm import tqdm

from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.generation import LogitsProcessorList
from trl import GRPOConfig
from trl.trainer.utils import pad

from constrained_logits_processor import (
    ConstrainedClusterLogitsProcessor,
    build_cluster_hash_dict,
    create_prefix_allowed_tokens_fn,
)


class RepeatRandomSampler(Sampler):
    """
    重复采样器：每个样本重复N次
    
    用于GRPO训练，确保每个prompt生成多个completions
    """
    
    def __init__(self, data_source, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)
    
    def __iter__(self):
        # 随机打乱，然后每个index重复repeat_count次
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)
    
    def __len__(self):
        return self.num_samples * self.repeat_count


class HierGRSeqRecGRPOTrainer(Trainer):
    """
    HierGR-SeqRec 的 GRPO 训练器
    
    核心功能：
    1. Constrained Beam Search 生成
    2. Group-wise reward 计算和归一化
    3. GRPO loss 计算（带KL正则化）
    4. 训练时评估
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        sid_mapping: dict,
        reward_funcs: Union[Callable, List[Callable]],
        args: GRPOConfig = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        # HierGR-SeqRec specific
        prompt2target: Optional[Dict[str, str]] = None,  # prompt -> target_cluster_str
        use_beam_search: bool = True,
        test_during_training: bool = True,
        test_beam: int = 20,
        **kwargs,
    ):
        # 初始化配置
        if args is None:
            args = GRPOConfig("hiergr-grpo")
        
        # 记录模型路径（用于后续加载 reference model）
        self.model_path = model if isinstance(model, str) else model.config._name_or_path
        
        # 先加载 tokenizer（从 SFT 模型路径，包含扩展的词汇表）
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side="left"
            )
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型（使用正确的词汇表大小）
        if isinstance(model, str):
            print(f"Loading model from {model}")
            print(f"Tokenizer vocab size: {len(tokenizer)}")
            
            # 检查是否有 adapter_config.json（表示这是 PEFT 模型）
            import os
            adapter_config_path = os.path.join(model, "adapter_config.json")
            has_adapter = os.path.exists(adapter_config_path)
            
            if has_adapter:
                print("Detected PEFT adapter, loading manually...")
                import json
                from safetensors import safe_open
                
                # 读取adapter配置获取base model路径
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-1.5B-Instruct')
                
                # 从adapter的safetensors中读取实际的vocab size
                adapter_weights_path = os.path.join(model, "adapter_model.safetensors")
                actual_vocab_size = None
                if os.path.exists(adapter_weights_path):
                    with safe_open(adapter_weights_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if "embed_tokens" in key and "weight" in key:
                                shape = f.get_tensor(key).shape
                                actual_vocab_size = shape[0]
                                print(f"Detected vocab size from adapter: {actual_vocab_size}")
                                break
                
                # 如果无法从adapter读取，使用tokenizer的大小
                if actual_vocab_size is None:
                    actual_vocab_size = len(tokenizer)
                    print(f"Using tokenizer vocab size: {actual_vocab_size}")
                
                print(f"Base model: {base_model_name}")
                print(f"Target vocab size: {actual_vocab_size}")
                
                # 步骤1: 加载base model（不加载adapter）
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    use_cache=False
                )
                
                # 步骤2: Resize embeddings到adapter的实际大小（在加载adapter之前）
                current_size = model.get_input_embeddings().weight.shape[0]
                if current_size != actual_vocab_size:
                    print(f"Resizing embeddings: {current_size} -> {actual_vocab_size}")
                    model.resize_token_embeddings(actual_vocab_size)
                
                # 步骤3: 加载PEFT adapter（此时embeddings大小已匹配）
                from peft import PeftModel
                print(f"Loading PEFT adapter from {self.model_path}")
                model = PeftModel.from_pretrained(model, self.model_path)
                
                # GRPO需要训练整个模型（包括adapter）
                # 确保所有adapter参数可训练
                model.train()
                for name, param in model.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                
                # 禁用gradient checkpointing（GRPO不需要，会导致问题）
                if hasattr(model, 'gradient_checkpointing_disable'):
                    model.gradient_checkpointing_disable()
                model.config.use_cache = False
                
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Trainable parameters: {trainable_params:,}")
            else:
                # 普通模型加载流程
                config = AutoConfig.from_pretrained(model)
                
                if hasattr(config, 'vocab_size') and config.vocab_size != len(tokenizer):
                    print(f"Resizing model embeddings: {config.vocab_size} -> {len(tokenizer)}")
                    config.vocab_size = len(tokenizer)
                
                model = AutoModelForCausalLM.from_pretrained(
                    model,
                    config=config,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                    use_cache=False
                )
                
                if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
                    print(f"Resizing token embeddings to {len(tokenizer)}")
                    model.resize_token_embeddings(len(tokenizer))
        
        # 存储配置
        self.sid_mapping = sid_mapping
        self.prompt2target = prompt2target or {}
        self.use_beam_search = use_beam_search
        self.test_during_training = test_during_training
        self.test_beam = test_beam
        
        self.num_generations = args.num_generations
        self.max_completion_length = args.max_completion_length
        self.beta = args.beta
        
        # Reward函数
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs
        
        # 构建约束哈希字典
        print("Building constrained generation hash dict...")
        self.hash_dict, self.prefix_index, self.get_hash = build_cluster_hash_dict(
            sid_mapping=sid_mapping,
            tokenizer=tokenizer,
            model_type=model.config.model_type
        )
        
        self.prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(
            self.hash_dict, 
            self.get_hash
        )
        
        # 生成配置
        if use_beam_search:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                num_beams=self.num_generations,
                num_return_sequences=self.num_generations,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 测试时的生成配置
        self.test_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            num_beams=test_beam,
            num_return_sequences=test_beam,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        # 指标存储
        self._metrics = defaultdict(list)
        
        # Data collator
        def data_collator(features):
            return features
        
        # 初始化父类
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs,
        )
        
        # 创建reference model（用于KL计算，从同一个 SFT 模型路径加载）
        print("Loading reference model...")
        import os
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        has_adapter = os.path.exists(adapter_config_path)
        
        if has_adapter:
            print("Loading reference model with PEFT adapter...")
            import json
            from safetensors import safe_open
            
            # 读取adapter配置
            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get('base_model_name_or_path', 'Qwen/Qwen2.5-1.5B-Instruct')
            
            # 从adapter的safetensors中读取实际的vocab size
            adapter_weights_path = os.path.join(self.model_path, "adapter_model.safetensors")
            actual_vocab_size = None
            if os.path.exists(adapter_weights_path):
                with safe_open(adapter_weights_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if "embed_tokens" in key and "weight" in key:
                            shape = f.get_tensor(key).shape
                            actual_vocab_size = shape[0]
                            print(f"Reference model: detected vocab size from adapter: {actual_vocab_size}")
                            break
            
            # 如果无法从adapter读取，使用tokenizer的大小
            if actual_vocab_size is None:
                actual_vocab_size = len(self.processing_class)
                print(f"Reference model: using tokenizer vocab size: {actual_vocab_size}")
            
            print(f"Reference model base: {base_model_name}")
            
            # 步骤1: 加载base model
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            
            # 步骤2: Resize embeddings到adapter的实际大小
            current_size = self.ref_model.get_input_embeddings().weight.shape[0]
            if current_size != actual_vocab_size:
                print(f"Resizing reference model embeddings: {current_size} -> {actual_vocab_size}")
                self.ref_model.resize_token_embeddings(actual_vocab_size)
            
            # 步骤3: 加载PEFT adapter
            from peft import PeftModel
            print(f"Loading reference PEFT adapter from {self.model_path}")
            self.ref_model = PeftModel.from_pretrained(self.ref_model, self.model_path)
            
            # Reference model不需要训练，禁用gradient checkpointing
            if hasattr(self.ref_model, 'gradient_checkpointing_disable'):
                self.ref_model.gradient_checkpointing_disable()
            self.ref_model.config.use_cache = False
        else:
            # 普通模型加载
            ref_config = AutoConfig.from_pretrained(self.model_path)
            
            if hasattr(ref_config, 'vocab_size') and ref_config.vocab_size != len(self.processing_class):
                print(f"Resizing reference model embeddings: {ref_config.vocab_size} -> {len(self.processing_class)}")
                ref_config.vocab_size = len(self.processing_class)
            
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=ref_config,
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            
            if self.ref_model.get_input_embeddings().weight.shape[0] != len(self.processing_class):
                print(f"Resizing reference model token embeddings to {len(self.processing_class)}")
                self.ref_model.resize_token_embeddings(len(self.processing_class))
        
        self.ref_model.eval()
        
        # 标记这个模型不接受loss kwargs
        self.model_accepts_loss_kwargs = False
    
    def _set_signature_columns_if_needed(self):
        """设置signature columns为 ["prompt"]"""
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        """返回重复采样器"""
        if train_dataset is None:
            train_dataset = self.train_dataset
        return RepeatRandomSampler(
            train_dataset, 
            self.num_generations, 
            seed=self.args.seed
        )
    
    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """返回重复采样器"""
        return RepeatRandomSampler(
            eval_dataset, 
            self.num_generations, 
            seed=self.args.seed
        )
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """计算每个token的log概率"""
        logits = model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        ).logits
        
        # logits[:, i] 预测的是 input_ids[:, i+1]
        # 所以需要对齐：logits[:, :-1] 对应 input_ids[:, 1:]
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        # 只保留completion部分（最后 logits_to_keep 个token）
        logits = logits[:, -logits_to_keep:]
        labels = labels[:, -logits_to_keep:]
        
        # 计算log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Gather the log probs for the actual tokens
        per_token_logps = torch.gather(
            log_probs, 
            2, 
            labels.unsqueeze(2)
        ).squeeze(2)
        
        return per_token_logps
    
    def _prepare_inputs(self, inputs: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        准备输入：生成completions并计算rewards
        
        这是GRPO的核心流程
        """
        device = self.accelerator.device
        
        # 提取prompts
        prompts = [x["prompt"] for x in inputs]
        
        # Tokenize prompts
        prompt_inputs = self.processing_class(
            prompts,
            return_tensors='pt',
            padding=True,
            padding_side="left",
            truncation=True,
            max_length=self.args.max_prompt_length
        )
        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs["attention_mask"].to(device)
        
        # 创建约束logits processor
        constrained_processor = ConstrainedClusterLogitsProcessor(
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            num_beams=self.num_generations if self.use_beam_search else 1,
            model_type=self.model.config.model_type
        )
        logits_processor = LogitsProcessorList([constrained_processor])
        
        # Debug: 验证约束处理器初始化
        # if self.state.global_step == 0:
        #     print(f"DEBUG: Created ConstrainedClusterLogitsProcessor with prompt_length={constrained_processor.prompt_length}")
        
        # 生成completions
        self.model.eval()
        
        # Debug: 打印生成配置
        # if self.state.global_step == 0:
        #     print(f"DEBUG: Generation config: {self.generation_config}")
        #     print(f"DEBUG: max_completion_length: {self.max_completion_length}")
        #     print(f"DEBUG: tokenizer.eos_token_id: {self.tokenizer.eos_token_id}")
        #     print(f"DEBUG: tokenizer.pad_token_id: {self.tokenizer.pad_token_id}")
        #     print(f"DEBUG: Sample prompt tokens: {prompts[0][:100]}")
        
        with torch.no_grad():
            if self.use_beam_search:
                # Beam search模式：去重后生成
                dedup_prompt_ids = prompt_ids[::self.num_generations]
                dedup_prompt_mask = prompt_mask[::self.num_generations]
                
                # if self.state.global_step == 0:
                #     print(f"DEBUG: Input to generate - dedup_prompt_ids shape: {dedup_prompt_ids.shape}")
                
                prompt_completion_ids = self.model.generate(
                    dedup_prompt_ids,
                    attention_mask=dedup_prompt_mask,
                    generation_config=self.generation_config,
                    logits_processor=logits_processor,
                )
                
                # if self.state.global_step == 0:
                #     print(f"DEBUG: Output from generate - shape: {prompt_completion_ids.shape}")
            else:
                # Sampling模式
                prompt_completion_ids = self.model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                    logits_processor=logits_processor,
                )
        
        self.model.train()
        
        # 分离prompt和completion
        if self.use_beam_search:
            # Beam search: 使用去重后的prompt长度
            prompt_length = dedup_prompt_ids.size(1)
        else:
            prompt_length = prompt_ids.size(1)
        
        # Debug信息（已关闭）
        # print(f"DEBUG: prompt_ids shape: {prompt_ids.shape}")
        # print(f"DEBUG: prompt_completion_ids shape: {prompt_completion_ids.shape}")
        # print(f"DEBUG: prompt_length: {prompt_length}")
        # print(f"DEBUG: use_beam_search: {self.use_beam_search}")
        # if self.use_beam_search:
        #     print(f"DEBUG: dedup_prompt_ids shape: {dedup_prompt_ids.shape}")
        
        completion_ids = prompt_completion_ids[:, prompt_length:]
        # print(f"DEBUG: completion_ids shape: {completion_ids.shape}")
        
        # Debug: 查看实际生成的内容（已关闭）
        # if self.state.global_step == 0:
        #     print(f"DEBUG: 第一个样本的completion tokens: {completion_ids[0].tolist()}")
        #     decoded = self.tokenizer.decode(completion_ids[0], skip_special_tokens=False)
        #     print(f"DEBUG: 第一个样本的completion文本: '{decoded}'")
        
        # 检查是否生成了任何token
        if completion_ids.size(1) == 0:
            raise RuntimeError(
                f"模型没有生成任何新token！\n"
                f"  - prompt长度: {prompt_length}\n"
                f"  - 生成输出长度: {prompt_completion_ids.size(1)}\n"
                f"  - max_new_tokens: {self.max_completion_length}\n"
                f"这通常是由于约束过于严格，导致所有token都被mask。\n"
                f"请检查：\n"
                f"  1. sid_mapping是否正确加载\n"
                f"  2. hash_dict是否正确构建\n"
                f"  3. prefix_allowed_tokens_fn是否返回有效token"
            )
        
        # Mask EOS之后的tokens
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        
        # 只对包含EOS的序列更新位置
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            eos_positions = is_eos.int().argmax(dim=1)
            eos_idx[has_eos] = eos_positions[has_eos]
        
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # 解码completions
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # 计算reference model的logprobs（用于KL散度）
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        with torch.no_grad():
            ref_per_token_logps = self._get_per_token_logps(
                self.ref_model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep
            )
        
        # 计算rewards
        rewards = torch.zeros(len(prompts), device=device)
        for reward_func in self.reward_funcs:
            output_rewards = reward_func(
                prompts=prompts,
                completions=completions_text,
                targets=[self.prompt2target.get(p, "") for p in prompts]
            )
            rewards += torch.tensor(output_rewards, dtype=torch.float32, device=device)
        
        # Group-wise normalization
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        
        # 计算advantages
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # 记录指标
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        
        # 训练时评估
        if self.test_during_training and self.state.global_step % self.args.logging_steps == 0:
            hr, ndcg = self._evaluate_during_training(prompt_ids, prompt_mask, prompts)
            for k, v in hr.items():
                self._metrics[f"HR@{k}"].append(v)
            for k, v in ndcg.items():
                self._metrics[f"NDCG@{k}"].append(v)
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    def _evaluate_during_training(self, prompt_ids, prompt_mask, prompts):
        """训练时进行快速评估"""
        device = prompt_ids.device
        
        # 去重prompts
        dedup_prompt_ids = prompt_ids[::self.num_generations]
        dedup_prompt_mask = prompt_mask[::self.num_generations]
        dedup_prompts = prompts[::self.num_generations]
        
        # 为测试创建专用的logits processor（beam size = test_beam）
        from transformers.generation import LogitsProcessorList
        test_processor = ConstrainedClusterLogitsProcessor(
            prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
            num_beams=self.test_beam,
            model_type=self.model.config.model_type
        )
        test_logits_processor = LogitsProcessorList([test_processor])
        
        # 生成
        with torch.no_grad():
            test_outputs = self.model.generate(
                dedup_prompt_ids,
                attention_mask=dedup_prompt_mask,
                generation_config=self.test_generation_config,
                logits_processor=test_logits_processor,
            )
        
        # 解码
        prompt_length = dedup_prompt_ids.size(1)
        test_completions = self.processing_class.batch_decode(
            test_outputs[:, prompt_length:],
            skip_special_tokens=True
        )
        
        # 计算HR和NDCG
        topk_list = [5, 10, 20]
        hr = {k: 0 for k in topk_list}
        ndcg = {k: 0 for k in topk_list}
        
        num_samples = len(dedup_prompts)
        test_comp_list = [
            test_completions[i:i+self.test_beam] 
            for i in range(0, len(test_completions), self.test_beam)
        ]
        
        for i, (prompt, completions) in enumerate(zip(dedup_prompts, test_comp_list)):
            target = self.prompt2target.get(prompt, "").strip()
            
            for j, completion in enumerate(completions):
                completion = completion.strip()
                if completion == target:
                    for k in topk_list:
                        if j < k:
                            hr[k] += 1
                            ndcg[k] += 1.0 / math.log2(j + 2)
                    break
        
        # 归一化
        hr = {k: v / num_samples for k, v in hr.items()}
        ndcg = {k: v / num_samples for k, v in ndcg.items()}
        
        return hr, ndcg
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算GRPO loss
        
        GRPO loss = -E[π(a|s) / π_old(a|s) * A(s,a)] + β * KL(π||π_ref)
        """
        if return_outputs:
            raise ValueError("HierGRSeqRecGRPOTrainer does not support returning outputs")
        
        # 确保model在训练模式
        model.train()
        
        # Debug: 检查model参数（已关闭）
        # if self.state.global_step == 0:
        #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #     total_params = sum(p.numel() for p in model.parameters())
        #     print(f"DEBUG: Model参数统计:")
        #     print(f"  - 可训练参数: {trainable_params:,}")
        #     print(f"  - 总参数: {total_params:,}")
        #     if trainable_params == 0:
        #         print(f"  - ⚠️ 警告：没有可训练参数！")
        
        # 获取输入
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]
        
        # 拼接input
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        # 计算当前策略的logprobs
        per_token_logps = self._get_per_token_logps(
            model, 
            input_ids, 
            attention_mask, 
            logits_to_keep
        )
        
        # Debug: 检查梯度状态（已关闭）
        # if self.state.global_step == 0:
        #     print(f"DEBUG compute_loss:")
        #     print(f"  - completion_ids shape: {completion_ids.shape}")
        #     print(f"  - completion_mask shape: {completion_mask.shape}")
        #     print(f"  - completion_mask sum per sample: {completion_mask.sum(dim=1)}")
        #     print(f"  - per_token_logps shape: {per_token_logps.shape}")
        #     print(f"  - per_token_logps requires_grad: {per_token_logps.requires_grad}")
        #     print(f"  - advantages shape: {advantages.shape}")
        #     print(f"  - ref_per_token_logps shape: {ref_per_token_logps.shape}")
        
        # 计算KL散度
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - \
                       (ref_per_token_logps - per_token_logps) - 1
        
        # 计算per-token loss
        # π/π_old = exp(log π - log π_old)
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        
        # 平均到sequence level
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # if self.state.global_step == 0:
        #     print(f"  - loss: {loss.item()}")
        #     print(f"  - loss requires_grad: {loss.requires_grad}")
        
        # 记录KL散度
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(mean_kl.item())
        
        return loss
    
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """记录指标"""
        # 计算平均指标
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items() if len(val) > 0}
        
        # 合并到logs
        logs = {**logs, **metrics}
        
        # 调用父类log
        super().log(logs, start_time)
        
        # 清空metrics
        self._metrics.clear()
