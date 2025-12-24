# 泛型类型变量
from typing import TypeVar, Generic, Optional

from pydantic import BaseModel

T = TypeVar('T')

class APIRes(BaseModel, Generic[T]):
    """
    通用API返回值格式

    Attributes:
        code: 错误码，200表示成功，非0表示失败
        message: 响应消息
        data: 响应数据，可选
    """
    code: int = 200
    message: str = "Success"
    data: Optional[T] = None


class PageParams(BaseModel):
    """
    分页请求参数基类

    Attributes:
        page: 页码，从1开始
        page_size: 每页大小
    """
    page: int = 1
    page_size: int = 10


class PageMeta(BaseModel):
    """
    分页元数据

    Attributes:
        page: 当前页码
        page_size: 每页大小
        total: 总记录数
        pages: 总页数
    """
    page: int
    page_size: int
    total: int


class PageRes(BaseModel, Generic[T]):
    """
    分页响应数据基类

    Attributes:
        items: 数据列表
        meta: 分页元数据
    """
    items: list[T]
    meta: PageMeta