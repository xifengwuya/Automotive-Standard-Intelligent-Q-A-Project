import uuid
import jieba
from rank_bm25 import BM25Okapi
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config.settings import settings
from modules.doc_processor import AutomotiveDocProcessor
import os
from transformers import BitsAndBytesConfig
import torch
from langchain_community.vectorstores.utils import filter_complex_metadata




class HybridRetrieval:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            # 检查向量数据库是否存在
            db_path = os.path.join(settings.VECTOR_DB_PATH, "chroma.sqlite3")
            if os.path.exists(db_path):
                # 加载现有数据库
                self._load_existing_database()
            else:
                # 初始化新数据库
                self._initialize_new_database()
            self._initialized = True

    def _initialize_new_database(self):
        """初始化新的数据库"""
        print("--初始化新的数据库--")
        # 1. 初始化文档处理器（获取汽车标准文本块）
        self.doc_processor = AutomotiveDocProcessor()
        self.text_chunks = []  # 文本块内容列表
        self.chunk_ids = []    # 文本块唯一ID列表
        self.chunk_metadata = []  # 文本块元数据（文件名、页码等）

        self._load_and_preprocess_docs()

        # 2. 构建BM25索引（中文需先分词）
        self.tokenized_corpus = [self._chinese_tokenize(chunk) for chunk in self.text_chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        if torch.cuda.is_available() :
            gpu_id = settings.TRAIN_GPU_IDS[0] if isinstance(settings.TRAIN_GPU_IDS, list) else settings.TRAIN_GPU_IDS
            torch.cuda.set_device(gpu_id)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu" ,
                "trust_remote_code": True,

            },
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": settings.BATCH_SIZE_PER_GPU
            }
        )

        # 修复：移除ids参数，先创建Chroma实例
        self.vector_db = Chroma(
            collection_name="automotive_standard",
            embedding_function=self.embeddings,
            persist_directory=settings.VECTOR_DB_PATH
        )

        # 修复：使用add_texts方法添加文档和ID
        if self.text_chunks:  # 确保有文本块才添加
            self.vector_db.add_texts(
                texts=self.text_chunks,
                ids=self.chunk_ids,
                metadatas=self.chunk_metadata
            )
            count = self.vector_db._collection.count()
            print(f"向量数据库中文档数量: {count}")
            if count == 0:
                print("警告：向量数据库为空")
                return []

    def _load_existing_database(self):
        """加载已存在的向量数据库"""
        print("--加载已存在的向量数据库--")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"}
        )
        self.vector_db = Chroma(
            collection_name="automotive_standard",
            embedding_function=self.embeddings,
            persist_directory=settings.VECTOR_DB_PATH
        )
        # 重新加载文档数据用于BM25检索
        self.doc_processor = AutomotiveDocProcessor()
        self.text_chunks = []
        self.chunk_ids = []
        self.chunk_metadata = []
        self._load_and_preprocess_docs()
        # 重新构建BM25索引
        self.tokenized_corpus = [self._chinese_tokenize(chunk) for chunk in self.text_chunks]
        # 在 HybridRetrieval 类的初始化方法中
        try:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except ZeroDivisionError:
            print("错误: 无法使用空语料库初始化 BM25 模型。")
            self.bm25 = None

    # def _load_docs_from_vector_db(self):
    #     """从向量数据库加载文档，避免重新处理PDF"""
    #     try:
    #         # 获取向量数据库中的所有文档
    #         collection = self.vector_db._collection
    #         if collection:
    #             # 获取所有文档
    #             results = collection.get(include=["documents", "metadatas"])
    #
    #             if results and "documents" in results:
    #                 self.text_chunks = results["documents"]
    #                 self.chunk_metadata = results.get("metadatas", [])
    #
    #                 # 重新生成chunk_ids
    #                 self.chunk_ids = []
    #                 for i, metadata in enumerate(self.chunk_metadata):
    #                     chunk_id = str(uuid.uuid4())
    #                     self.chunk_ids.append(chunk_id)
    #                     # 确保metadata中包含chunk_id
    #                     if metadata and isinstance(metadata, dict):
    #                         metadata["chunk_id"] = chunk_id
    #
    #                 print(f"从向量数据库加载了 {len(self.text_chunks)} 个文档块")
    #             else:
    #                 print("向量数据库中没有文档")
    #     except Exception as e:
    #         print(f"从向量数据库加载文档失败: {e}")
    #         self.text_chunks = []
    #         self.chunk_ids = []
    #         self.chunk_metadata = []
    #
    def _filter_metadata(self, metadata):
        """
        过滤元数据，只保留Chroma支持的数据类型
        """
        if not isinstance(metadata, dict):
            return {}

        filtered = {}
        for key, value in metadata.items():
            # 只保留支持的数据类型：str, int, float, bool, None
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value
            elif isinstance(value, list):
                # 对于列表，转换为字符串存储
                filtered[key] = str(value)
            elif isinstance(value, dict):
                # 对于嵌套字典，递归过滤
                filtered[key] = self._filter_metadata(value)
            else:
                # 不支持的类型转换为字符串
                filtered[key] = str(value)

        return filtered

    def _load_and_preprocess_docs(self):
        """加载汽车标准PDF，预处理为文本块并分配唯一ID"""
        train_pdf_paths = [f"{settings.TRAIN_DOC_PATH}/{f}" for f in os.listdir(settings.TRAIN_DOC_PATH) if f.endswith(".pdf")]
        # test: 使用VAL_DOC_PATH
        # train_pdf_paths = [f"{settings.VAL_DOC_PATH}/{f}" for f in os.listdir(settings.VAL_DOC_PATH) if
        #                    f.endswith(".pdf")]
        for pdf_path in train_pdf_paths:
            raw_docs = self.doc_processor.load_pdf_doc(pdf_path)  # 加载PDF
            split_docs = self.doc_processor.text_splitter.split_documents(raw_docs)
            # print(f"已处理文件：{raw_docs}")
            # print(split_docs[0])
            for doc in split_docs:
                chunk_id = str(uuid.uuid4())
                self.chunk_ids.append(chunk_id)
                self.text_chunks.append(doc.page_content)
                # 过滤元数据后再添加
                filtered_metadata = self._filter_metadata(doc.metadata)
                self.chunk_metadata.append(filtered_metadata)

    def _chinese_tokenize(self, text):
        """中文分词（适配BM25，去除停用词）"""
        stop_words = set(["的", "了", "是", "在", "和", "有"])  # 可扩展停用词表
        tokens = jieba.lcut(text)
        return [token for token in tokens if token not in stop_words and len(token) > 1]

    def retrieve_bm25(self, query, top_k=10):
        """BM25检索"""
        tokenized_query = self._chinese_tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)  # 获取所有文本块得分
        # 按得分排序，取Top-K
        top_indices = bm25_scores.argsort()[::-1][:top_k]
        # 构造结果：(chunk_id, text, metadata, bm25_score)
        bm25_results = [
            (
                self.chunk_ids[idx],
                self.text_chunks[idx],
                self.chunk_metadata[idx],
                bm25_scores[idx]
            )
            for idx in top_indices
        ]
        return bm25_results

    def retrieve_vector(self, query, k=settings.RETRIEVE_TOP_K):
        """向量检索（沿用原有逻辑）"""
        # 添加调试信息
        if not hasattr(self, 'vector_db') or self.vector_db is None:
            print("错误：向量数据库未正确初始化")
            return []

        # 检查数据库中是否有文档
        try:
            count = self.vector_db._collection.count()
            print(f"向量数据库中文档数量: {count}")
            if count == 0:
                print("警告：向量数据库为空")
                return []
        except Exception as e:
            print(f"检查数据库文档数量时出错: {e}")
        vector_docs = self.vector_db.similarity_search_with_score(query, k=k)
        # 构造结果：(chunk_id, text, metadata, vector_score)
        vector_results = [
            (
                doc[0].id,
                doc[0].page_content,
                doc[0].metadata,
                doc[1]  # 余弦相似度得分
            )
            for doc in vector_docs
        ]
        return vector_results

    def hybrid_fusion(self, query, top_k=10, alpha=0.6):
        """
        后期融合核心逻辑
        :param query: 用户查询
        :param top_k: 最终返回的候选集大小
        :param alpha: 向量得分权重（BM25得分权重=1-alpha），建议0.5-0.7（汽车行业可设0.6，兼顾语义与专业术语）
        :return: 融合排序后的候选集
        """
        # 1. 分别获取BM25和向量检索结果
        bm25_res = self.retrieve_bm25(query, top_k =top_k * 2)  # 取2倍Top-K，确保足够候选
        vector_res = self.retrieve_vector(query, k =top_k * 2)

        # 2. 转换为字典（以chunk_id为键，方便合并）
        bm25_dict = {res[0]: (res[1], res[2], res[3]) for res in bm25_res}
        vector_dict = {res[0]: (res[1], res[2], res[3]) for res in vector_res}

        # 3. 合并结果（去重，包含所有在BM25或向量检索中的chunk）
        all_chunk_ids = set(bm25_dict.keys()).union(set(vector_dict.keys()))
        merged_results = []
        for chunk_id in all_chunk_ids:
            # 获取BM25得分（无则设为0）
            bm25_score = bm25_dict[chunk_id][2] if chunk_id in bm25_dict else 0.0
            # 获取向量得分（无则设为0）
            vector_score = vector_dict[chunk_id][2] if chunk_id in vector_dict else 0.0

            # 4. 加权计算最终得分（归一化后再加权，避免得分范围差异影响结果）
            # 归一化BM25得分（0-1）
            max_bm25 = max([r[3] for r in bm25_res]) if bm25_res else 1.0
            norm_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
            # 归一化向量得分（余弦相似度本身在0-1之间，无需额外归一化）
            norm_vector = vector_score

            # 最终得分 = alpha*归一化向量得分 + (1-alpha)*归一化BM25得分
            final_score = alpha * norm_vector + (1 - alpha) * norm_bm25

            # 构造合并结果 - 安全获取文本和元数据
            if chunk_id in bm25_dict:
                text, metadata = bm25_dict[chunk_id][0], bm25_dict[chunk_id][1]
            elif chunk_id in vector_dict:
                text, metadata = vector_dict[chunk_id][0], vector_dict[chunk_id][1]
            else:
                # 理论上不会到达这里，但为了安全起见
                continue

            merged_results.append((chunk_id, text, metadata, final_score))

        # 5. 按最终得分排序，取Top-K
        merged_results.sort(key=lambda x: x[3], reverse=True)
        final_results = merged_results[:top_k]

        # 6. 转换为LangChain文档格式（适配后续BERT二次打分和LLM输入）
        from langchain_core.documents import Document
        langchain_docs = [
            Document(
                page_content=res[1],
                metadata={
                    **res[2],
                    "chunk_id": res[0],
                    "final_score": res[3],
                    "file_name": res[2].get("file_name", "")  # 确保filename字段存在
                }
            )
            for res in final_results
        ]
        return langchain_docs
hybrid_retrieval_instance = HybridRetrieval()
