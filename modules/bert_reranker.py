import torch
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from langchain_core.documents import Document
from config.settings import settings
import os


class AutomotiveBertReranker:
    def __init__(self):
        # 优先加载微调后模型，否则加载原始模型
        model_path = settings.SAVE_MODEL_PATH if os.path.exists(settings.SAVE_MODEL_PATH) else settings.BERT_MODEL_NAME
        if os.path.exists(os.path.join(model_path, "model.safetensors")):
            model_path = os.path.join(model_path, "model.safetensors")
        elif os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
            model_path = os.path.join(model_path, "pytorch_model.bin")
        else:
            # 从HuggingFace在线加载
            model_path = "hfl/chinese-bert-wwm-ext"

        self.tokenizer = BertTokenizer.from_pretrained(model_path)

        # 加载模型（支持多GPU推理）
        self.model = BertModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 多GPU封装（DP模式，简单高效）
        if torch.cuda.device_count() > 1 and len(settings.TRAIN_GPU_IDS) > 1:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=settings.TRAIN_GPU_IDS
            )

        self.model.to(self.device)
        self.model.eval()

    def _get_text_embedding(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        # 处理多GPU输出（DP模式下若批次为1，需挤压维度）
        if cls_embedding.ndim > 1:
            cls_embedding = cls_embedding.mean(axis=0)
        return cls_embedding / np.linalg.norm(cls_embedding)

    def rerank_docs(self, query: str, initial_docs: List[Document]) -> List[Document]:
        if not initial_docs:
            return []

        query_embedding = self._get_text_embedding(query)
        doc_embeddings = [self._get_text_embedding(doc.page_content) for doc in initial_docs]

        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        reranked_docs = [initial_docs[i] for i in sorted_indices[:settings.RERANK_TOP_N]]

        for idx, doc in enumerate(reranked_docs):
            doc.metadata["bert_similarity_score"] = float(similarities[sorted_indices[idx]])

        return reranked_docs


bert_reranker = AutomotiveBertReranker()