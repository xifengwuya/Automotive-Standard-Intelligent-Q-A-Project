from typing import Generator, Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session

from app.core.config import settings
from app.core.db import engine
from app.crud.users_crud import get_user_by_sub, UserIn
from app.models.token_models import TokenPayload

# 定义 OAuth2 密码Bearer 依赖项
# 用于在路由 /access-token 中获取访问令牌
reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/login/access-token"
)

TokenDep = Annotated[str, Depends(reusable_oauth2)]


def get_db() -> Generator[Session, None, None]:
    """
    数据库会话依赖项
    用于在 FastAPI 路由中获取数据库会话，确保在请求处理完成后自动关闭会话

    Yields:
        Session: 数据库会话实例
    """
    with Session(engine) as session:
        try:
            yield session
        except SQLAlchemyError:
            session.rollback()
        finally:
            session.close()


SessionDep = Annotated[Session, Depends(get_db)]


def get_current_user(session: SessionDep, token: TokenDep) -> UserIn:
    """
    获取当前用户
    用于在 FastAPI 路由中获取当前认证用户，根据 JWT 令牌中的 subject (sub) 字段查询数据库

    Args:
        session (SessionDep): 数据库会话依赖项
        token (TokenDep): JWT 令牌依赖项

    Returns:
        UserIn: 当前认证用户对象

    Raises:
        HTTPException: 403 权限拒绝，验证失败或用户不存在
        HTTPException: 404 用户不存在，数据库查询结果为空
        HTTPException: 400 禁用用户，用户状态为 False
    """
    try:
        # 解码 JWT 令牌
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_data = TokenPayload(**payload)
    except (InvalidTokenError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    # 从数据库中查询用户
    user = get_user_by_sub(session=session, sub=token_data.sub)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user.status:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


CurrentUser = Annotated[UserIn, Depends(get_current_user)]
