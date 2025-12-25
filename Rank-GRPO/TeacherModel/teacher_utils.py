# teacher_utils.py
import json
import numpy as np

class TeacherOracle:
    def __init__(self, teacher_prediction_file):
        """
        teacher_prediction_file: 一个 JSON 文件，格式为:
        {
            "user_id_or_history_hash": ["gmap_id_1", "gmap_id_2", ..., "gmap_id_50"],
            ...
        }
        这里存储了 SASRec 认为用户接下来最可能去的 Top-N 个地点。
        """
        print(f"👨‍🏫 Loading Teacher Predictions from {teacher_prediction_file}...")
        with open(teacher_prediction_file, 'r') as f:
            self.predictions = json.load(f)
            
    def get_top_k(self, user_key, k=10, exclude_set=None):
        """
        获取教师模型的推荐列表，用于 SFT 填充或 RL 奖励计算
        """
        candidates = self.predictions.get(str(user_key), [])
        if exclude_set:
            candidates = [c for c in candidates if c not in exclude_set]
        return candidates[:k]

    def get_rank(self, user_key, target_gmap_id):
        """
        RL 阶段使用：获取生成的物品在教师模型中的排名
        """
        candidates = self.predictions.get(str(user_key), [])
        try:
            # 返回 rank (0-based)
            return candidates.index(target_gmap_id)
        except ValueError:
            # 如果没在 Top-N 里，返回一个很大的惩罚值
            return len(candidates) + 100