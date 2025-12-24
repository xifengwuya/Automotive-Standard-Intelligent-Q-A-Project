from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm

from app.api.deps import SessionDep
from app.core import security
from app.core.config import settings
from app.crud import users_crud
from app.models.token_models import Token

router = APIRouter(tags=["login"])

@router.post("/login/access-token")
def login_access_token(
    session: SessionDep, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    """
    OAuth2 与 Jwt 登录接口，用于获取访问令牌

    该函数处理用户登录请求，验证用户名和密码，检查用户状态，
    并生成JWT访问令牌

    Args:
        session: 数据库会话依赖项，用于数据库操作
        form_data: OAuth2密码请求表单数据，包含用户名和密码

    Returns:
        Token: 包含访问令牌的Token模型实例

    Raises:
        HTTPException: 当用户名或密码错误时抛出400状态码异常
        HTTPException: 当用户处于非活跃状态时抛出400状态码异常
    """
    # 验证用户凭据
    user = users_crud.authenticate(
        session=session, username=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.status:
        raise HTTPException(status_code=400, detail="Inactive user")

    # 设置访问令牌过期时间
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    # 创建并返回包含访问令牌的Token对象
    return Token(
        access_token=security.create_access_token(
            str(user.id), expires_delta=access_token_expires
        )
    )
