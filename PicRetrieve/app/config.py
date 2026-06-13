"""运行配置。"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """! @brief PicRetrieve 的环境变量配置。"""

    image_root: Path = Field(default=Path("samples"), alias="PICRETRIEVE_IMAGE_ROOT")
    data_dir: Path = Field(default=Path("data"), alias="PICRETRIEVE_DATA_DIR")
    model_name: str = Field(
        default="data/models/chinese-clip-vit-base-patch16", alias="PICRETRIEVE_MODEL_NAME"
    )
    device: str | None = Field(default=None, alias="PICRETRIEVE_DEVICE")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


def load_settings() -> Settings:
    """! @brief 从环境变量和 `.env` 加载配置。"""

    return Settings()


def resolve_image_root(settings: Settings) -> Path:
    """! @brief 获取 API 允许访问的图片根目录。

    如果索引命令写入了 `image_root.txt`，优先使用该路径，避免 API 只能服务
    默认 `samples/` 的图片。
    """

    marker = settings.data_dir / "image_root.txt"
    if marker.exists():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return Path(value).expanduser().resolve()
    return settings.image_root.expanduser().resolve()
