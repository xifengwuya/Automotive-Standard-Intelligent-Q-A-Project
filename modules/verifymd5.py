import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from config.settings import settings
import pandas as pd

test_queries_df = pd.read_csv(settings.TEST_QUERY_PATH, encoding="utf-8-sig")
print(test_queries_df.head())

#
# def calculate_md5(file_path):
#     hash_md5 = hashlib.md5()
#     with open(file_path, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5.update(chunk)
#     return hash_md5.hexdigest()
#
#
# file_path = r"D:\aaaCS\automotive_rag_qa\models\Qwen2.5_VL\model-00002-of-00002.safetensors"
#
# import hashlib
#
# def calculate_sha256(file_path):
#     hash_sha256 = hashlib.sha256()
#     with open(file_path, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_sha256.update(chunk)
#     return hash_sha256.hexdigest()
#
# sha256_hash2 = calculate_sha256(file_path)
# if sha256_hash2 == "365531ff8752420e89dee707b79d021fb2d6e25abafe486f080555a4fe6972e4":
#     print("embedding模型文件验证成功")
# else:
#     print("embedding模型文件验证失败")


