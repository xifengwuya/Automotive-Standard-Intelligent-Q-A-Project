from sqlalchemy.ext.declarative import declarative_base
from sqlmodel import create_engine

from app.core.config import settings

# 创建数据库引擎
engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI_MYSQL), echo=settings.DEBUG_MYSQL)

# 创建基础模型类
Base = declarative_base()

# 导入所有模型类，确保它们被注册到 Base 中 这样 create_all() 才能找到并创建所有的表
# 避免无关的警告，如未使用的导入、行太长等 noqa
from app.models import users_models  # noqa

def create_tables():
    """创建数据库表
    检查数据库是否有表，没有则创建，有则不做处理
    """
    try:
        Base.metadata.create_all(bind=engine)
        print("数据库表检查完成: 已存在的表保持不变，不存在的表已创建")
    except Exception as e:
        print(f"数据库表创建失败: {e}")
        raise

def shutdown_db():
    """关闭数据库
    删除全部会话，关闭引擎
    """
    try:
        # 关闭引擎，释放所有连接池资源
        engine.dispose()
        print("数据库已关闭: 所有会话已删除，引擎已关闭")
    except Exception as e:
        print(f"数据库关闭失败: {e}")
        raise
