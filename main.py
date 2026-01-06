import argparse
import uvicorn
from fastapi import FastAPI, Query
from pydantic import BaseModel

from modules.rag_chain import automotive_rag_chain
from evaluations.test_runner import EvaluationRunner
import asyncio


# 初始化FastAPI应用
app = FastAPI(title="汽车行业标准RAG智能问答系统")


# 响应模型
class QAResponse(BaseModel):
    user_query: str
    answer: str
    sources: list
    retrieved_context: str = None


@app.get("/qa", response_model=QAResponse)
async def qa_endpoint(query: str = Query(..., description="汽车行业标准查询问题")):
    """智能问答接口"""
    result = automotive_rag_chain.run(query)
    return QAResponse(
        user_query=result["user_query"],
        answer=result["answer"],
        sources=result["sources"],
        retrieved_context=result["retrieved_context"]
    )


def main():
    parser = argparse.ArgumentParser(description="汽车行业标准RAG智能问答系统")
    parser.add_argument("--mode", type=str, choices=["api", "eval"], default="eval",
                        help="运行模式：api（启动接口服务），eval（执行评估测试）")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API服务地址")
    parser.add_argument("--port", type=int, default=8000, help="API服务端口")

    args = parser.parse_args()

    if args.mode == "eval":
        # 执行评估测试
        eval_runner = EvaluationRunner()
        eval_runner.run_evaluation()
    elif args.mode == "api":
        # 启动FastAPI服务
        if args.mode == "api":
            # 启动FastAPI服务
            asyncio.run(uvicorn.run(app, host=args.host, port=args.port, lifespan="on"))


if __name__ == "__main__":
    main()