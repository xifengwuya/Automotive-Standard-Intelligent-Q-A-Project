import os
from typing import List
from dotenv import load_dotenv
import torch

class Settings:
    # 数据路径配置
    PDF_ROOT_PATH = "./data/pdf_docs"
    TRAIN_DOC_PATH = "./data/train_docs"
    VAL_DOC_PATH = "./data/val_docs"
    TEST_QUERY_PATH = "./data/test_queries/automotive_test_queries.csv"
    VECTOR_DB_PATH = "./models/chroma_automotive_db"
    BERT_MODEL_NAME = "./models/hfl/chinese-bert-wwm-ext"
    EMBEDDING_MODEL_NAME = "./models/BAAI/bge-small-zh-v1.5"  # 轻量中文嵌入模型
    # BERT_MODEL_NAME = "/home/wanglihua/automotive_rag_qa/models/chinese-bert-wwm-ext"
    # EMBEDDING_MODEL_NAME = "/home/wanglihua/automotive_rag_qa/models/BAAI/bge-small-zh-v1.5"  # 轻量中文嵌入模型
    # LLM配置（使用本地模型/API，此处以HuggingFace本地模型为例）
    LLM_MODEL_NAME = "./models/Qwen2.5-1.5B"  # "THUDM/chatglm3-6b"
    # LLM_MODEL_NAME = "/home/wanglihua/automotive_rag_qa/models/Qwen2.5-1.5B"
    # 分块配置（适配汽车行业标准结构化文档）
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 100
    SEPARATORS: List[str] = ["\n\n", "\n", "。", "；", "，", "（", "）", "、"]

    # 检索配置
    RETRIEVE_TOP_K = 15  # 初始检索候选数
    RERANK_TOP_N = 5  # BERT二次打分后最终保留数

    # 评估配置
    BASELINE_MODEL_NAMES = ["pure_vector_rag", "no_rerank_rag"]  # 基线模型名称
    EVAL_METRICS = ["hit_rate@1", "hit_rate@3", "mrr", "bleu-2", "rouge-l"]  # 评估指标
    LLM_MAX_LENGTH = 2048
    TEMPERATURE = 0.1  # 生成更严谨（适配行业标准问答）

    # ===================== 新增：多GPU训练配置 =====================
    TRAIN_GPU_IDS = [0]  # 指定使用的GPU编号（如[0,1]表示使用第1、2张GPU）
    USE_DDP = False  # 是否使用DDP模式（False=DP模式，True=DDP模式）
    BATCH_SIZE_PER_GPU = 32  # 每张GPU的批次大小
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    WEIGHcdT_DECAY = 0.01
    SAVE_MODEL_PATH = "./models/finetuned_bert_automotive"  # 微调后模型保存路径
    TRAIN_DATA_PATH = "./data/bert_finetune_data"  # BERT微调数据集路径
    DASHSCOPE_API_KEY = os.getenv("API_KEY")
    query = 100
    HF_HUB_DISABLE_SYMLINKS_WARNING = 1
    IMAGE_SAVE_PATH = "./data/output_images"
    HF_TOKEN = os.getenv("HF_TOKEN")
    QWEN_VL_MODEL_PATH = "./models/Qwen2.5_VL"
    CROSS_ENCODER_MODEL_PATH = "./models/cross-encoder/ms-marco-MiniLM-L-6-v2"


settings = Settings()

# 创建必要目录
os.makedirs(settings.TRAIN_DOC_PATH, exist_ok=True)
os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(settings.TEST_QUERY_PATH), exist_ok=True)

# 验证GPU可用性
# if torch.cuda.is_available():
#     assert len(settings.TRAIN_GPU_IDS) <= torch.cuda.device_count(), f"可用GPU数量为{torch.cuda.device_count()}，少于配置的{len(settings.TRAIN_GPU_IDS)}张"
# else:
#     raise EnvironmentError("当前环境不支持CUDA，无法进行多GPU训练")