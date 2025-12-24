from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router
from app.core.config import settings
from app.core.db import create_tables, shutdown_db


def custom_generate_unique_id(route: APIRoute) -> str:
    """
    自定义API路由唯一ID生成函数

    该函数用于为FastAPI路由生成唯一的标识符，格式为"{标签}-{路由名称}"，
    便于在API文档和路由管理中识别不同的端点。

    Args:
        route (APIRoute): FastAPI路由对象

    Returns:
        str: 格式化的唯一ID字符串，格式为"{route.tags[0]}-{route.name}"
    """
    return f"{route.tags[0]}-{route.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 应用启动时执行
    print("应用启动中...")
    create_tables()  # 调用同步函数
    print("应用启动完成")
    yield
    # 应用关闭时执行
    print("应用关闭中...")
    shutdown_db()  # 调用同步函数
    print("应用关闭完成")


# 创建FastAPI应用实例，配置项目基本信息和生命周期管理
app = FastAPI(
    title=settings.PROJECT_NAME,  # API 文档标题，来自配置文件
    openapi_url=f"{settings.API_V1_STR}/openapi.json",  # OpenAPI 文档的 URL 路径
    generate_unique_id_function=custom_generate_unique_id,  # 自定义路由 ID 生成函数
    lifespan=lifespan,  # 使用生命周期管理
)

# 根据配置启用CORS中间件，允许跨域请求
if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,  # 允许的来源列表，来自配置文件
        allow_credentials=True,  # 允许携带凭证（如 cookies）
        allow_methods=["*"],  # 允许所有 HTTP 方法
        allow_headers=["*"],  # 允许所有 HTTP 头部
    )

# 包含API路由器，设置路由前缀
app.include_router(api_router, prefix=settings.API_V1_STR)

