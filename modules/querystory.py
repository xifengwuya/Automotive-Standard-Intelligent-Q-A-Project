import os
import torch
from dotenv import load_dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
# 关键：导入HuggingFacePipeline封装本地模型
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 加载环境变量（仅Redis配置，无需LLM API Key）
load_dotenv()


# -------------------------- 1. 加载本地Qwen2.5-1.5B模型 --------------------------
def load_local_qwen_model(model_path: str = "./models/Qwen2.5-1.5B") -> HuggingFacePipeline:
    """
    加载本地部署的Qwen2.5-1.5B模型，封装为LangChain可用的Pipeline
    """
    # 1. 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,  # Qwen需要开启
        padding_side="right"  # 右填充，避免推理错误
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # GPU用float16，CPU用float32
        device_map="auto"  # 自动分配设备（有GPU用GPU，无则用CPU）
    )

    # 2. 创建推理Pipeline（适配对话场景）
    qwen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # 推理参数（适配Qwen2.5-1.5B，可根据需求调整）
        max_new_tokens=512,  # 生成最大长度
        temperature=0.1,  # 低温度保证回答严谨
        top_p=0.9,  # 采样策略
        repetition_penalty=1.1,  # 避免重复生成
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        return_full_text=False  # 只返回生成的内容，不包含输入
    )

    # 3. 封装为LangChain的LLM对象
    llm = HuggingFacePipeline(
        pipeline=qwen_pipeline,
        model_kwargs={"temperature": 0.1}  # 覆盖温度参数（可选）
    )
    return llm


# -------------------------- 2. 初始化Redis对话历史（不变） --------------------------
def init_redis_chat_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(
        session_id=session_id,
        redis_url=f"redis://{os.getenv('REDIS_PASSWORD') or ''}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}/{os.getenv('REDIS_DB')}",
        ttl=3600 * 24,  # 24小时过期
    )


# -------------------------- 3. 初始化对话记忆（不变） --------------------------
def init_conversation_memory(session_id: str, k: int = 5) -> ConversationBufferWindowMemory:
    chat_history = init_redis_chat_history(session_id)
    memory = ConversationBufferWindowMemory(
        chat_memory=chat_history,
        k=k,
        return_messages=True,
        memory_key="history",
        input_key="input"
    )
    return memory


# -------------------------- 4. 初始化多轮对话链（适配本地Qwen） --------------------------
def init_automotive_conversation_chain(session_id: str) -> ConversationChain:
    """
    初始化汽车行业多轮对话链（本地Qwen2.5-1.5B + Redis记忆）
    """
    # 加载本地Qwen模型
    llm = load_local_qwen_model("./models/Qwen2.5-1.5B")

    # 汽车行业专用Prompt（适配Qwen2.5的对话格式）
    automotive_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""你是资深汽车行业标准解读专家，仅基于对话历史和当前问题回答，严谨准确。
        对话历史：{history}
        当前问题：{input}
        回答要求：
        1. 引用具体汽车标准编号（如GB 7258-2017）；
        2. 技术参数必须准确，禁止编造；
        3. 多轮对话中需关联历史问题；
        4. 无法回答时明确说明“无相关标准依据”。
        回答："""
    )

    # 初始化记忆组件
    memory = init_conversation_memory(session_id, k=5)

    # 构建对话链（本地模型无需verbose=True，减少日志冗余）
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=automotive_prompt,
        verbose=False  # 本地模型推理日志较多，关闭详细输出
    )
    return conversation_chain


# -------------------------- 5. 测试多轮对话 --------------------------
if __name__ == "__main__":
    # 1. 配置Redis环境变量（如果.env没配置，手动设置）
    os.environ.setdefault("REDIS_HOST", "localhost")
    os.environ.setdefault("REDIS_PORT", "6379")
    os.environ.setdefault("REDIS_DB", "0")

    # 2. 指定用户ID
    USER_ID = "automotive_user_001"
    # 3. 初始化对话链（首次加载模型可能需要1-2分钟，耐心等待）
    print("正在加载本地Qwen2.5-1.5B模型...")
    chain = init_automotive_conversation_chain(USER_ID)
    print("模型加载完成，开始多轮对话：\n")

    # 4. 多轮对话示例
    # 第1轮
    response1 = chain.invoke({"input": "乘用车50km/h制动距离限值是多少？"})
    print(f"AI回答1：{response1['response'].strip()}\n")

    # 第2轮（追问，关联历史）
    response2 = chain.invoke({"input": "货车的对应标准是什么？"})
    print(f"AI回答2：{response2['response'].strip()}\n")

    # 第3轮（模拟重启程序，验证Redis持久化）
    print("===== 模拟重启程序，加载Redis历史 =====")
    new_chain = init_automotive_conversation_chain(USER_ID)
    response3 = new_chain.invoke({"input": "刚才提到的货车标准编号是多少？"})
    print(f"AI回答3：{response3['response'].strip()}\n")

    # 可选：清理对话历史
    # chat_history = init_redis_chat_history(USER_ID)
    # chat_history.clear()
    # print("已清理该用户的对话历史")