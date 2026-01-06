import os
import sys

import requests

# 将项目根目录添加到路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shutil
import random
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import Settings
from .pdf_processor import MultimodalPDFProcessor


class AutomotiveDocProcessor:
    def __init__(self):
        # 初始化分块器（适配汽车行业标准条款式结构）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP,
            separators=Settings.SEPARATORS,
            length_function=len
        )
        self.pdf_processor = MultimodalPDFProcessor()

    def load_pdf_doc(self, pdf_path: str) -> List[Document]:
        """加载单个PDF文档并进行结构化分块"""
        if not pdf_path.endswith(".pdf"):
            raise ValueError(f"仅支持PDF文件，当前文件：{pdf_path}")
        text_chunks = self.pdf_processor.process_pdf(pdf_path)
        return text_chunks

    def _read_pdf_content(self, pdf_path: str) -> str:
        """读取PDF文档的完整内容"""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        # 将所有页面内容合并
        content = ""
        for doc in documents:
            content += doc.page_content
        # PyPDFLoader 通常会自动处理资源，但确保清理
        return content


    def split_train_test_docs(self, test_size: float = 1, random_seed: int = 42) -> None:
        """
        将PDF文档划分为训练集和测试集
        :param test_size: 测试集比例（此处测试集为文档级，用于评估检索泛化性）
        :param random_seed: 随机种子，保证划分可复现
        """
        # 获取所有PDF文档路径
        all_pdf_paths = [
            os.path.join(Settings.PDF_ROOT_PATH, fname)
            for fname in os.listdir(Settings.PDF_ROOT_PATH)
            if fname.endswith(".pdf")
        ]
        # if len(all_pdf_paths) < 5:
        #     raise ValueError("PDF文档数量过少，至少需要5个文档用于划分")

        # 随机划分
        random.seed(random_seed)
        random.shuffle(all_pdf_paths)
        test_doc_num = int(len(all_pdf_paths) * test_size)
        train_pdf_paths = all_pdf_paths[test_doc_num:]
        test_pdf_paths = all_pdf_paths[:test_doc_num]

        # 复制训练集文档到训练目录
        for pdf_path in train_pdf_paths:
            shutil.copy(pdf_path, os.path.join(Settings.TRAIN_DOC_PATH, os.path.basename(pdf_path)))

        # 生成测试查询集
        # 如果文件已经存在，不用生成
        if not os.path.exists(Settings.TEST_QUERY_PATH):
            self._generate_test_queries(test_pdf_paths)
        print(f"数据集划分完成：")
        print(f"  训练集文档数：{len(train_pdf_paths)}，存储路径：{Settings.TRAIN_DOC_PATH}")
        print(f"  测试集文档数：{test_doc_num}，测试查询集已生成：{Settings.TEST_QUERY_PATH}")

    def _generate_test_queries(self, test_pdf_paths: List[str]) -> None:
        """生成测试查询集（通过阿里百炼平台API生成2000个问答对）"""
        import pandas as pd

        # 配置阿里百炼API参数
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        api_key = Settings.DASHSCOPE_API_KEY

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        test_queries = []
        total_pairs_needed = 2000
        pairs_generated = 0

        # 循环处理每个PDF文档，直到生成足够的问答对
        while pairs_generated < total_pairs_needed:
            for pdf_path in test_pdf_paths:
                if pairs_generated >= total_pairs_needed:
                    break

                # 读取PDF文档内容用于生成问答对
                doc_content = self._read_pdf_content(pdf_path)

                # 构建生成问答对的提示（每次生成5个问答对）
                prompt = f"""
                基于以下汽车标准文档内容，请生成5个相关的问答对，用于测试RAG系统。
                文档内容: {doc_content[:2000]}

                请按照以下格式返回:
                问题1: [问题内容]
                答案1: [答案内容]

                问题2: [问题内容]
                答案2: [答案内容]
                """

                payload = {
                    "model": "qwen-plus",
                    "input": {
                        "prompt": prompt
                    },
                    "parameters": {
                        "temperature": 0.7,
                        "max_tokens": 1024
                    }
                }

                try:
                    response = requests.post(api_url, headers=headers, json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        # 解析API返回的问答对
                        qa_pairs = self._parse_qa_pairs(result["output"]["text"])

                        # 添加到测试查询列表
                        for j, qa_pair in enumerate(qa_pairs):
                            if pairs_generated < total_pairs_needed:
                                test_queries.append({
                                    "query_id": len(test_queries) + 1,
                                    "query": qa_pair["question"],
                                    "ground_truth_answer": qa_pair["answer"],
                                    "ground_truth_doc": os.path.basename(pdf_path),
                                    "query_type": self._classify_query_type(qa_pair["question"])
                                })
                                pairs_generated += 1
                            else:
                                break
                    else:
                        print(f"API调用失败: {response.status_code}, {response.text}")
                        # 备用方案：生成模板查询
                        self._generate_template_queries(test_queries, pdf_path, 0)
                except Exception as e:
                    print(f"API调用异常: {e}")
                    # 备用方案：生成模板查询
                    self._generate_template_queries(test_queries, pdf_path, 0)

        # 保存为CSV文件
        df = pd.DataFrame(test_queries)
        df.to_csv(Settings.TEST_QUERY_PATH, index=False, encoding="utf-8-sig")

    def _parse_qa_pairs(self, text: str) -> List[dict]:
        """解析API返回的问答对文本"""
        qa_pairs = []
        lines = text.split('\n')
        current_question = None
        current_answer = None

        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            if line.startswith('问题') and ':' in line:
                if current_question and current_answer:
                    qa_pairs.append({
                        "question": current_question.strip(),
                        "answer": current_answer.strip()
                    })
                # 安全地分割并检查长度
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    current_question = parts[1].strip()
                else:
                    current_question = ""  # 设置为空字符串而不是跳过，保持问答对的完整性
                current_answer = None
            elif line.startswith('答案') and ':' in line:
                # 安全地分割并检查长度
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    current_answer = parts[1].strip()
                else:
                    current_answer = ""

        # 添加最后一对问答（如果存在）
        if current_question is not None and current_answer is not None:
            if current_question or current_answer:  # 确保不是空的问答对
                qa_pairs.append({
                    "question": current_question.strip(),
                    "answer": current_answer.strip()
                })

        return qa_pairs

    def _classify_query_type(self, question: str) -> str:
        """分类查询类型"""
        # 简单的关键词分类
        if any(keyword in question.lower() for keyword in ['定义', '是什么', '含义']):
            return 'definition'
        elif any(keyword in question.lower() for keyword in ['如何', '怎样', '步骤', '方法']):
            return 'procedure'
        elif any(keyword in question.lower() for keyword in ['要求', '标准', '规定', '规范']):
            return 'requirement'
        else:
            return 'general'

    def _generate_template_queries(self, test_queries: List[dict], pdf_path: str, index: int):
        """生成模板查询作为备用方案"""
        base_questions = [
            f"关于{os.path.basename(pdf_path)}文档的主要内容是什么？",
            f"{os.path.basename(pdf_path)}文档中提到了哪些关键要求？",
            f"根据{os.path.basename(pdf_path)}文档，相关标准是如何规定的？",
            f"{os.path.basename(pdf_path)}文档的核心要点有哪些？",
            f"从{os.path.basename(pdf_path)}文档中可以了解到什么信息？"
        ]

        for j, question in enumerate(base_questions):
            test_queries.append({
                "query_id": len(test_queries) + 1,
                "query": question,
                "ground_truth_answer": f"这是{os.path.basename(pdf_path)}文档的模板答案",
                "ground_truth_doc": os.path.basename(pdf_path),
                "query_type": "general"
            })


# 初始化并执行数据集划分（首次运行时执行）
if __name__ == "__main__":
    processor = AutomotiveDocProcessor()
    processor.split_train_test_docs(test_size=0.2)
