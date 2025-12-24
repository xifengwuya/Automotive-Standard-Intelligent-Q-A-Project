from datetime import datetime

from sqlalchemy import String, DateTime, Boolean, BigInteger, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class UserModel(Base):
    __tablename__ = "user_t"
    __table_args__ = (
        {
            'comment': '用户表',
            'mysql_charset': 'utf8mb4',
            'mysql_collate': 'utf8mb4_0900_ai_ci'
        }
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True, comment="用户id")
    username: Mapped[str] = mapped_column(String(50), nullable=False, unique=True, comment="用户名")
    password: Mapped[str] = mapped_column(String(100), nullable=False, comment="密码")
    nickname: Mapped[str] = mapped_column(String(50), nullable=True, comment="昵称")
    email: Mapped[str] = mapped_column(String(100), nullable=True, comment="邮箱")
    phone: Mapped[str] = mapped_column(String(20), nullable=True, comment="手机号")
    avatar: Mapped[str] = mapped_column(String(500), nullable=True, comment="头像")
    intro: Mapped[str] = mapped_column(String(500), nullable=True, comment="个人简介")
    role_id: Mapped[int] = mapped_column(BigInteger, nullable=True, comment="角色id")
    status: Mapped[bool] = mapped_column(Boolean, default=True, comment="状态 0-禁用 1-启用")
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, comment="逻辑删除 0-未删除 1-已删除")
    create_time: Mapped[datetime] = mapped_column(DateTime, nullable=True, server_default=text('CURRENT_TIMESTAMP'),
                                                  comment="创建时间")
    update_time: Mapped[datetime] = mapped_column(DateTime, nullable=True,
                                                  server_default=text('CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP'),
                                                  onupdate=text('CURRENT_TIMESTAMP'), comment="更新时间")
