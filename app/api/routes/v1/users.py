from fastapi import APIRouter, HTTPException, status

from app.api.deps import SessionDep, CurrentUser
from app.crud.users_crud import UserCreate, get_user_by_username, create_user, UserIn
from app.models.base_models import APIRes

router = APIRouter(prefix="/users", tags=["users"])

@router.post("/register", response_model=APIRes[bool])
def register_user(*, session: SessionDep, user: UserCreate):
    """
    注册新用户

    Args:
        session: 数据库会话
        user: 用户注册信息

    Returns:
        bool: 注册成功返回True，否则返回False
    """

    # 检查用户名是否已存在
    existing_user = get_user_by_username(session=session, username=user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # 创建新用户
    res = create_user(session=session, user=user) is not None
    return APIRes(data=res, message="User registered successfully")

@router.get("/me", response_model=APIRes[UserIn])
def get_current_user(*, current_user: CurrentUser):
    """
    获取当前登录用户信息

    Args:
        current_user: 当前登录用户

    Returns:
        UserIn: 当前登录用户信息
    """
    return APIRes(data=current_user, message="User info retrieved successfully")
