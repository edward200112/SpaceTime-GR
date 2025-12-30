import json
from typing import Dict, List, Tuple, Optional

def norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    s = (s.replace("’", "'")
           .replace("“", '"').replace("”", '"')
           .replace("–", "-").replace("—", "-"))
    s = " ".join(s.split())
    return s

def parse_namecat(text: str) -> Tuple[str, str]:
    """
    解析 "Name (Category)".
    若失败，cat 返回 ""，name 返回原文本（归一化后）。
    """
    t = (text or "").strip()
    if "(" in t and t.endswith(")"):
        name = t.rsplit("(", 1)[0].strip()
        cat = t.rsplit("(", 1)[1][:-1].strip()
        return norm(name), norm(cat)
    return norm(t), ""

class NameCatResolver:
    def __init__(
        self,
        namecat2item_unique_path: str,
        namecat2item_disamb_path: str,
        name2item_disamb_path: str,
        sasrec_model=None,   # 传入你冻结的 SASRec
        device="cuda",
    ):
        self.device = device
        self.sasrec = sasrec_model

        with open(namecat2item_unique_path, "r", encoding="utf-8") as f:
            self.namecat2item_unique: Dict[str, List[int]] = json.load(f)
        with open(namecat2item_disamb_path, "r", encoding="utf-8") as f:
            self.namecat2item_disamb: Dict[str, List[int]] = json.load(f)
        with open(name2item_disamb_path, "r", encoding="utf-8") as f:
            self.name2item_disamb: Dict[str, List[int]] = json.load(f)

    def resolve_candidates(self, pred_text: str) -> List[int]:
        name, cat = parse_namecat(pred_text)
        if not name:
            return []

        # 1) namecat unique 命中
        key = f"{name} ({cat})" if cat else ""
        if key and key in self.namecat2item_unique:
            return self.namecat2item_unique[key]

        # 2) namecat disamb 命中
        if key and key in self.namecat2item_disamb:
            return self.namecat2item_disamb[key]

        # 3) category 不可信/Place/写错 -> name-only fallback
        # 你现在大量 miss 都来自这里
        name_key = name
        if name_key in self.name2item_disamb:
            return self.name2item_disamb[name_key]

        return []

    @staticmethod
    def _is_bad_cat(cat: str) -> bool:
        c = (cat or "").strip().lower()
        return (c == "" or c == "place" or c == "poi" or c == "unknown")

    def pick_best_with_sasrec(self, history_item_ids: List[int], candidate_item_ids: List[int]) -> Optional[int]:
        """
        用 SASRec 在候选里挑最符合 history 的那个 item。
        """
        if not candidate_item_ids:
            return None
        if len(candidate_item_ids) == 1 or self.sasrec is None:
            return int(candidate_item_ids[0])

        import torch
        self.sasrec.eval()

        # input_ids: [1, L]
        inp = torch.tensor([history_item_ids], dtype=torch.long, device=self.device)
        cand = torch.tensor([candidate_item_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            scores = self.sasrec.predict_candidates(inp, cand)  # [1, C]
            best_idx = int(scores[0].argmax().item())
        return int(candidate_item_ids[best_idx])

    def resolve_to_item_id(self, pred_text: str, history_item_ids: List[int]) -> Optional[int]:
        cands = self.resolve_candidates(pred_text)
        if not cands:
            return None
        return self.pick_best_with_sasrec(history_item_ids, cands)
