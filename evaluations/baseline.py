from typing import Dict, Any, List
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from config.settings import settings
from modules.rag_chain import AutomotiveRAGChain
from modules.hybrid_retrieval import HybridRetrieval
class Doc:
    """文档对象，包含metadata属性"""
    def __init__(self, metadata: Dict[str, Any], page_content: str):
        self.metadata = metadata
        self.page_content = page_content

class BaselineModels:
    def __init__(self):
        self.rag_core = AutomotiveRAGChain()  # 复用LLM和提示词模板

    def pure_vector_rag(self, query: str) -> Dict[str, Any]:
        """基线1：纯向量检索（无LLM生成，仅返回Top-N初始检索文档）"""
        initial_docs = HybridRetrieval().retrieve_vector(query)

        top_n_docs = initial_docs[:settings.RERANK_TOP_N]

        return {
            "user_query": query,
            "retrieved_docs": [
                {
                    "file_name":  doc[2].get("file_name", ""),
                    "content": doc[1],
                    "final_score": doc[3],
                }
                for doc in top_n_docs
            ],
            "answer": "纯向量检索基线：无生成回答，仅返回检索文档"
        }

    def no_rerank_rag(self, query: str) -> Dict[str, Any]:
        """基线2：二次打分的RAG（初始向量检索+LLM生成）"""
        # 仅初始检索，不进行二次重排序
        initial_docs = HybridRetrieval().retrieve_vector(query)
        top_n_docs = initial_docs[:settings.RERANK_TOP_N]

        # 构建上下文
        context = "\n\n".join([
            f"【文档来源：{doc[2].get("file_name", "")}，相似度：{doc[3]:.4f}】\n{doc[1]}"
            for doc in top_n_docs
        ])

        # LLM生成回答
        response = self.rag_core.rag_chain.invoke({"context": context, "query": query})

        # 处理响应
        if isinstance(response, dict):
            answer_text = response.get("text", str(response))
        else:
            answer_text = str(response)

        return {
            "user_query": query,
            "retrieved_context": context,
            "answer": answer_text.strip(),
            "sources": [
                {
                    "file_name": doc[2].get("file_name", ""),
                    "content": doc[1],
                    "similarity_score": doc[3]
                }
                for doc in top_n_docs
            ]
        }

# 初始化基线模型
baseline_models = BaselineModels()