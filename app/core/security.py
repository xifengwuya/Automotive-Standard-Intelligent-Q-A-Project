from datetime import timedelta, timezone, datetime

import jwt
from pwdlib import PasswordHash

from app.core.config import settings

# 密码哈希算法
password_hash = PasswordHash.recommended()

def verify_password(plain_password, hashed_password):
    """
    验证密码是否匹配
    :param plain_password: 明文密码
    :param hashed_password: 存储的密码哈希值
    :return: 如果密码匹配则返回 True，否则返回 False
    """
    return password_hash.verify(plain_password, hashed_password)

def get_password_hash(password):
    """
    获取密码哈希值
    :param password: 明文密码
    :return: 密码哈希值
    """
    return password_hash.hash(password)

def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """
    创建访问令牌
    :param subject: 包含用户信息的字典
    :param expires_delta: 过期时间，默认值为 None
    :return: 编码后的 JWT 字符串
    """
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"exp": expire, "sub": subject}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt