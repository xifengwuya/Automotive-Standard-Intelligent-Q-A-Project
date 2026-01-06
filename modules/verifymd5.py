import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from config.settings import settings

# # 测试BERT模型
# try:
#     tokenizer = BertTokenizer.from_pretrained(settings.BERT_MODEL_NAME)
#     model = BertModel.from_pretrained(settings.BERT_MODEL_NAME)
#     print("BERT模型加载成功")
# except Exception as e:
#     print(f"BERT模型加载失败: {e}")
#
# # 测试嵌入模型
# try:
#     embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
#     print("嵌入模型加载成功")
# except Exception as e:
#     print(f"嵌入模型加载失败: {e}")
#
# import json
#
# # 检查配置文件是否可读
# with open('/home/wanglihua/automotive_rag_qa/models/chinese-bert-wwm-ext/config.json', 'r') as f:
#     config = json.load(f)
#     print(f"BERT 模型配置: {config.get('model_type', 'Unknown')}")
#
# with open('/home/wanglihua/automotive_rag_qa/models/bge-small-zh-v1.5/config.json', 'r') as f:
#     config = json.load(f)
#     print(f"嵌入模型配置: {config.get('model_type', 'Unknown')}")

import hashlib

def calculate_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 验证模型文件
# file_path = "/home/wanglihua/automotive_rag_qa/models/Qwen2.5-1.5B/model.safetensors"
file_path = r"D:\aaaCS\automotive_rag_qa\models\Qwen2.5_VL\model-00002-of-00002.safetensors"

import hashlib

def calculate_sha256(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# 验证文件
# sha256_hash = calculate_sha256(file_path)
# if sha256_hash == "a961db72e75d52b18e6b0c9d379e51a26973b233385e0e127fdda7d648aec796":
#     print("qwen模型文件验证成功")
# else:
#     print("qwen模型文件验证失败")
sha256_hash2 = calculate_sha256(file_path)
if sha256_hash2 == "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4":
    print("embedding模型文件验证成功")
else:
    print("embedding模型文件验证失败")


