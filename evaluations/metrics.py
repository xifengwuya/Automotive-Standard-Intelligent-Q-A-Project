import numpy as np
from typing import List, Dict, Any
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from langchain_core.documents import Document


class EvaluationMetrics:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        self.smoother = SmoothingFunction().method1  # BLEU分数平滑处理

    def calculate_retrieval_metrics(self, retrieved_docs: List[Document], ground_truth_doc: str) -> Dict[str, float]:
        """
        计算检索指标（Hit Rate@k、MRR）
        :param retrieved_docs: 检索到的文档列表（按相似度排序）
        :param ground_truth_doc: 真实相关文档名
        :return: 检索指标字典
        """
        # 提取检索文档的文件名
        retrieved_doc_names = [doc.metadata["file_name"] for doc in retrieved_docs]

        # Hit Rate@1：第一名是否是真实文档
        hit_rate_1 = 1.0 if len(retrieved_doc_names) > 0 and retrieved_doc_names[0] == ground_truth_doc else 0.0

        # Hit Rate@3：前3名是否包含真实文档
        hit_rate_3 = 1.0 if ground_truth_doc in retrieved_doc_names[:3] else 0.0

        # MRR（平均倒数排名）：真实文档的排名倒数，无则为0
        try:
            rank = retrieved_doc_names.index(ground_truth_doc) + 1  # 排名从1开始
            mrr = 1.0 / rank
        except ValueError:
            mrr = 0.0

        return {
            "hit_rate@1": hit_rate_1,
            "hit_rate@3": hit_rate_3,
            "mrr": mrr
        }

    def calculate_generation_metrics(self, generated_answer: str, ground_truth_answer: str) -> Dict[str, float]:
        """
        计算生成质量指标（BLEU-2、ROUGE-L）
        :param generated_answer: 模型生成的回答
        :param ground_truth_answer: 真实标注答案
        :return: 生成指标字典
        """
        # 预处理文本（分词）
        generated_tokens = generated_answer.split()
        ground_truth_tokens = [ground_truth_answer.split()]  # BLEU要求真实答案是二维列表

        # BLEU-2分数
        bleu_2 = sentence_bleu(
            ground_truth_tokens,
            generated_tokens,
            weights=(0.5, 0.5),
            smoothing_function=self.smoother
        )

        # ROUGE-L分数
        rouge_scores = self.rouge_scorer.score(ground_truth_answer, generated_answer)
        rouge_l = rouge_scores["rougeL"].fmeasure

        return {
            "bleu-2": bleu_2,
            "rouge-l": rouge_l
        }

    def compute_average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """计算批量指标的平均值"""
        avg_metrics = {}
        for metric_name in metrics_list[0].keys():
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in metrics_list])
        return avg_metrics


# 初始化评估指标计算器
eval_metrics = EvaluationMetrics()