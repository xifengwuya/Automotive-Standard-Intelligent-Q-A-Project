from datetime import datetime

import torch
from dotenv import load_dotenv
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import base64
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import os
from config.settings import settings
from transformers import BitsAndBytesConfig
from langchain_core.documents import Document
from typing import List


class MultimodalPDFProcessor:
    def __init__(self):  # 使用正确的模型名称
        try:
            # 初始化处理器与模型
            load_dotenv()
            hf_token = settings.HF_TOKEN
            qwen_vl_model = settings.QWEN_VL_MODEL_PATH
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            self.processor = AutoProcessor.from_pretrained(
                qwen_vl_model, use_fast=True)
            # 使用4位量化加载模型（更节省内存）
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                 qwen_vl_model,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
                device_map="auto",
                token=hf_token
            )
            self.model = self.model.eval()  # 设置为评估模式
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model {qwen_vl_model}: {str(e)}")

    def process_pdf(self, pdf_path) -> List[Document]:
        """处理PDF文件，提取所有页面的语义信息"""
        if not pdf_path or not os.path.exists(pdf_path):
            raise ValueError(f"PDF path does not exist: {pdf_path}")
        # 从环境变量获取图片保存路径
        image_save_path = settings.IMAGE_SAVE_PATH
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        try:
            # 获取原始PDF文件名（不含路径）
            original_pdf_name = os.path.basename(pdf_path)
            #检查image_save_path是否存在pdf_path同名文件夹
            image_folder = os.path.join(image_save_path, os.path.splitext(os.path.basename(pdf_path))[0])
            if os.path.exists(image_folder):
                print(f"Directory {image_folder} already exists. Skipping PDF to image conversion.")
                try:
                    image_path = [f for f in os.listdir(image_folder) if f.endswith('.png')]
                    image_paths = [os.path.join(image_folder, f) for f in image_path]
                except FileNotFoundError:
                    raise FileNotFoundError(f"Directory {image_folder} does not exist")
                except PermissionError:
                    raise PermissionError(f"No permission to access directory {image_folder}")
                except OSError as e:
                    raise OSError(f"Error accessing directory {image_folder}: {e}")
            else:
                print(f"Converting PDF to images...")
                image_paths = self.pdf_to_images(pdf_path, image_folder)

            all_semantic_chunks = []

            for page_num, image_path in enumerate(image_paths):
                try:
                    # 获取模型分析的语义块
                    print(f"Processing page {page_num + 1}...")
                    semantic_chunks_for_page = self.analyze_page_layout(image_path,original_pdf_name)
                    all_semantic_chunks.extend(semantic_chunks_for_page)
                    print(f"Successfully processed page {page_num + 1}")


                except Exception as e:
                    print(f"Warning: Failed to process page {page_num + 1}: {str(e)}")
                    continue

            return all_semantic_chunks

        except Exception as e:
            raise RuntimeError(f"Failed to process PDF {pdf_path}: {str(e)}")

    def pdf_to_images(self, pdf_path, output_folder, zoom=3):
        """
        将PDF的每一页转换为高质量图像，返回图像路径列表[3](@ref)
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        pdf_document = None
        try:
            pdf_document = fitz.open(pdf_path)
            image_paths = []

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)

                output_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
                pix.save(output_path)
                image_paths.append(output_path)

            print(f"转换完成！共生成 {len(image_paths)} 张图片。")
            return image_paths
        finally:
            if pdf_document:
                pdf_document.close()

    def ocr_with_layout(self, pdf_path):
        """使用OCR提取PDF文本，保留基本布局信息[4](@ref)"""
        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            for page_num in range(doc.page_count):
                page = doc[page_num]
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # 使用Tesseract进行OCR
                extracted_text = pytesseract.image_to_string(image, lang='chi_sim+eng')  # 支持中英文
                full_text += f"--- Page {page_num + 1} ---\n{extracted_text}\n"

            doc.close()
            return full_text

        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {str(e)}")

    def analyze_page_layout(self, image_path, original_pdf_name= None):

        """分析页面图像，识别布局元素并生成包含语义摘要的Document对象列表"""
        prompt = """
        请严格遵循以下JSON格式输出分析结果，仅返回JSON数据：
        {
            "sections": [
                {
                    "type": "元素类型（如：标题、段落、列表、表格、图表等）",
                    "content": "该元素的文本内容或语义摘要",
                    "bbox_2d": [x1, y1, x2, y2]  // 可选的边界框坐标
                }
            ]
        }

        请分析此PDF页面图像，识别出所有重要的布局元素。
        """

        try:

            # 构建符合模型要求的消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",  # 指向本地图像文件
                        },
                        {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ]

            # 使用工具包处理视觉信息
            image_inputs, video_inputs = process_vision_info(messages)

            # 应用聊天模板并处理文本输入
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 准备模型的完整输入
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)

            # 模型推理，生成回答
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

            # 解码模型输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            # 核心修改：解析响应并构造Document对象列表
            documents = self._parse_response_to_documents(response_text, image_path, original_pdf_name)
            return documents

        except Exception as e:
            # 打印完整的错误轨迹，这对于诊断至关重要
            import traceback
            error_detail = traceback.format_exc()
            print(f"分析页面布局失败 {image_path}。详细错误信息:\n{error_detail}")
            raise RuntimeError(f"分析页面布局失败 {image_path}: {str(e)}")


    def _parse_response_to_documents(self, response_text, image_path, original_pdf_name= None):
            """将模型响应解析为Document对象列表"""
            documents = []

            try:
                # 尝试解析JSON响应
                import json
                # 改进的JSON提取逻辑
                cleaned_text = self._extract_json_from_response(response_text)
                if cleaned_text:
                    response_data = json.loads(cleaned_text)
                    file_name = original_pdf_name if original_pdf_name else image_path

                    # 确保包含sections字段
                    if "sections" not in response_data:
                        # 如果响应不是预期JSON格式，将整个响应作为一个文档处理
                        return [Document(
                            page_content=response_text,
                            metadata={
                                "file_name": file_name,
                                "analysis_timestamp": datetime.now().isoformat(),
                                "section_type": "full_page_analysis",
                                "parser": "fallback"
                            }
                        )]

                    # 解析每个识别出的区块
                    for i, section in enumerate(response_data["sections"]):
                        # 创建Document对象
                        doc = Document(
                            page_content=section.get("content", ""),
                            metadata={
                                "file_name": file_name,
                                "section_type": section.get("type", "unknown"),
                                "section_order": i,
                                "bbox_2d": section.get("bbox_2d", []),
                                "analysis_timestamp": datetime.now().isoformat(),
                                "parser": "structured"
                            }
                        )
                        documents.append(doc)

            except json.JSONDecodeError:
                # 如果JSON解析失败，回退到将整个响应作为单个文档
                file_name = original_pdf_name if original_pdf_name else image_path
                documents.append(Document(
                    page_content=response_text,
                    metadata={
                        "file_name": file_name,
                        "analysis_timestamp": datetime.now().isoformat(),
                        "section_type": "full_page_fallback",
                        "parser": "fallback"
                    }
                ))

            return documents

    def _extract_json_from_response(self, response_text):
        """从模型响应中提取有效的JSON内容"""
        import re
        import json

        # 尝试提取Markdown代码块中的JSON
        pattern = r'(?:json)?\s*({.?})\s'
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # 尝试提取花括号内的内容
        pattern = r'\{.*\}'
        matches = re.findall(pattern, response_text, re.DOTALL)
        if matches:
            # 尝试找到最外层的完整JSON对象
            for match in matches:
                try:
                    json.loads(match.strip())
                    return match.strip()
                except json.JSONDecodeError:
                    continue

        # 如果没有找到匹配项，返回原始内容
        return response_text.strip()


# 测试代码
if __name__ == "__main__":
    try:
        pdf_processor = MultimodalPDFProcessor()
        # 使用示例PDF路径
        pdf_path = r"D:\aaaCS\automotive_rag_qa\data\pdf_docs\GBT+39631-2020.pdf"

        # 检查文件是否存在
        if os.path.exists(pdf_path):
            print("开始处理PDF...")
            result = pdf_processor.process_pdf(pdf_path)
            print("处理完成！")
            for i, chunk in enumerate(result):
                print(f"语义块 {i + 1}: {chunk[:200]}...")  # 只打印前200字符
        else:
            print(f"PDF文件不存在: {pdf_path}")

    except Exception as e:
        print(f"程序执行出错: {e}")