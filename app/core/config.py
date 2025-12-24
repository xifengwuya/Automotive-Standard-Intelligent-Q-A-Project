import secrets
from typing import Any, Literal, Annotated

from pydantic import AnyUrl, BeforeValidator, computed_field
from pydantic_settings import SettingsConfigDict, BaseSettings


def parse_cors(v: Any) -> list[str] | str:
    """
    CORS 来源解析函数
    将环境变量中的 CORS 来源字符串转换为列表格式

    参数:
        v: 输入值，可以是字符串或列表
        - 如果是逗号分隔的字符串（非 JSON 格式），则转换为去重后的列表
        - 如果是 JSON 格式字符串或列表，则直接返回
        - 其他格式则抛出 ValueError

    返回:
        list[str] | str: 解析后的 CORS 来源列表或字符串
    """
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",") if i.strip()]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)

class Settings(BaseSettings):
    """
    应用程序配置类
    从环境变量加载配置，支持 .env 文件
    """

    model_config = SettingsConfigDict(
        # 使用项目根目录下的 .env 文件
        env_file=".env",
        # 忽略空的环境变量值
        env_ignore_empty=True,
        # 忽略额外的环境变量（不在此类中定义的）
        extra="ignore",
    )

    # API 版本前缀
    API_V1_STR: str = "/api/v1"

    # JWT 令牌加密密钥，可被环境变量覆盖
    # secrets.token_urlsafe 用于生成安全的随机字符串，32 字节长度
    SECRET_KEY: str = secrets.token_urlsafe(32)

    # JWT 令牌加密算法
    ALGORITHM: str = "HS256"

    # 访问令牌过期时间（分钟）
    # 60分钟 * 24小时 * 8天 = 8天
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8

    # 运行环境
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"

    # 前端主机地址
    FRONTEND_HOST: str

    # 允许的 CORS 来源列表
    # 支持逗号分隔字符串格式或 JSON 数组格式
    BACKEND_CORS_ORIGINS: Annotated[
        list[AnyUrl] | str, BeforeValidator(parse_cors)
    ] = []

    # @computed_field 用于计算属性，计算属性的值会根据其他属性动态计算
    @computed_field  # type: ignore[prop-decorator]
    # @property 装饰器用于定义只读属性
    @property
    def all_cors_origins(self) -> list[str]:
        """
        获取所有允许的 CORS 来源
        合并 BACKEND_CORS_ORIGINS 和 FRONTEND_HOST

        返回:
            list[str]: 所有允许的 CORS 来源列表
        """
        return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
            self.FRONTEND_HOST
        ]

    # 项目名称
    # Pydantic Settings 会自动将类属性名转换为环境变量名（保持大写）
    # 支持从多来源加载配置，优先级为：系统环境变量 > 指定的 .env 文件 > 默认值
    # 通过 env_ignore_empty=True 配置，会忽略空的环境变量值
    PROJECT_NAME: str

    # MySQL 配置
    MYSQL_SERVER: str
    MYSQL_PORT: int = 3306
    MYSQL_USER: str
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def SQLALCHEMY_DATABASE_URI_MYSQL(self) -> str:
        """生成 MySQL 数据库连接 URL"""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_SERVER}:{self.MYSQL_PORT}/{self.MYSQL_DB}?charset=utf8mb4"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DEBUG_MYSQL(self) -> bool:
        """是否在本地环境下启用 MySQL 调试模式"""
        if self.ENVIRONMENT == "local":
            return True
        return False


# 创建全局配置对象实例
# 所有其他模块可以通过导入此实例来访问配置
# 手动忽略类型检查器的误报 type: ignore
settings = Settings() # type: ignore