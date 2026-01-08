from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from config.settings import settings
from modules.hybrid_retrieval import HybridRetrieval
from transformers import BitsAndBytesConfig


class AutomotiveRAGChain:
    def __init__(self):
        # 本地初始化LLM管道
        self.llm = self._init_llm()
        # 构建RAG提示词模板（适配汽车行业标准问答，强调严谨性）
        self.cross_encoder = CrossEncoder(settings.CROSS_ENCODER_MODEL_PATH)
        # self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
        self.prompt_template = PromptTemplate(
            template="""
            你是汽车行业标准智能问答助手，需基于以下提供的汽车行业标准文档内容，严谨、准确地回答用户问题。
            若文档内容不足以回答问题，请明确说明“根据现有标准文档，无法回答该问题”，禁止编造信息。

            相关汽车行业标准文档内容：
            {context}

            用户问题：{query}

            回答要求：
            1. 严格遵循文档内容，保持专业、严谨；
            2. 分点说明（若涉及多条条款），条理清晰；
            3. 注明引用文档来源（文档名）。
            """,
            input_variables=["context", "query"]
        )

        self.rag_chain = self.prompt_template | self.llm
    def _init_llm(self) -> HuggingFacePipeline:
        """初始化本地LLM管道"""
        tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME, trust_remote_code=True)
        # 替换原有的量化参数
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_NAME,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,  # 使用新的配置方式
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.eval()

        # 构建生成管道
        generate_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.LLM_MAX_LENGTH,
            temperature=settings.TEMPERATURE,
            top_p=0.9,
            repetition_penalty=1.1
        )

        return HuggingFacePipeline(pipeline=generate_pipeline)

    def run(self, query: str) -> Dict[str, Any]:
        """
        执行RAG检索生成流程
        :param query: 用户查询
        :return: 包含查询、上下文、回答、来源的字典
        """
        hybrid_docs = HybridRetrieval().hybrid_fusion(query, top_k=settings.RETRIEVE_TOP_K)
        print(f"检索结果:{hybrid_docs[0:5]}")
        print("使用cross encoder进行重排序...")
        reranked_docs = self.cross_encoder_rerank(query, hybrid_docs)

        # 步骤3：构建上下文
        context = "\n\n".join([
            f"【文档来源：{doc.metadata['file_name']}，相似度：{doc.metadata.get('cross_score', 0):.4f}】\n{doc.page_content}"
            for doc in reranked_docs
        ])

        # 步骤4：LLM生成回答
        response = self.rag_chain.invoke({"context": context, "query": query})
        #
        answer_text = response["text"] if isinstance(response, dict) else str(response)
        # 整理结果
        return {
            "user_query": query,
            "retrieved_context": context,
            "answer": answer_text.strip(),
            "sources": [
                {
                    "file_name": doc.metadata["file_name"],
                    "similarity_score": doc.metadata.get("cross_score", 0),
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                }
                for doc in reranked_docs
            ]
        }

    def cross_encoder_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """使用cross encoder进行重排序"""
        if not docs:
            return []

        # 构建查询-文档对
        query_doc_pairs = [(query, doc.page_content) for doc in docs]

        # 计算相关性得分
        scores = self.cross_encoder.predict(query_doc_pairs)

        # 将得分添加到文档元数据
        for i, doc in enumerate(docs):
            doc.metadata["cross_score"] = float(scores[i])

        # 按得分排序
        sorted_docs = sorted(docs, key=lambda x: x.metadata["cross_score"], reverse=True)

        return sorted_docs


# 初始化RAG链（全局单例）
automotive_rag_chain = AutomotiveRAGChain()