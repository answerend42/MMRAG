"""LLM 客户端封装。

支持 OpenAI 兼容 API，可通过环境变量配置：
- MRAG_LLM_API_KEY
- MRAG_LLM_BASE_URL
- MRAG_LLM_MODEL
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class LLMClient:
    """轻量 LLM 客户端，包装 OpenAI 兼容 API。

    WARNING: 本客户端假设 API 密钥已通过环境变量或参数注入。
    生产环境中不得在日志中打印完整请求体或密钥。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.model = model or os.getenv("MRAG_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.getenv("MRAG_LLM_API_KEY", "")
        self.base_url = base_url or os.getenv("MRAG_LLM_BASE_URL", "")
        self.temperature = temperature

        if not self.api_key:
            logger.warning(
                "MRAG_LLM_API_KEY is not set — LLM calls will likely fail. "
                "Set it in .env or environment variables."
            )

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """调用 LLM 并返回文本响应。

        Args:
            messages: OpenAI 格式的消息列表。
            **kwargs: 透传给 chat.completions.create 的额外参数。

        Returns:
            模型输出的文本内容。

        Raises:
            RuntimeError: API 调用失败时抛出。
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for LLMClient. Install with: uv add openai"
            ) from None

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.pop("temperature", self.temperature),
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM API call failed: {exc}") from exc

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned empty response")
        return content

    def chat_structured(
        self, messages: list[dict[str, Any]], response_model: type, **kwargs: Any
    ) -> Any:
        """调用 LLM 并解析为 Pydantic 模型（结构输出）。

        优先使用 OpenAI 的 structured output（response_format），
        降级到 JSON 解析。
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required for LLMClient") from None

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        try:
            response = client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_model,
                temperature=kwargs.pop("temperature", self.temperature),
                **kwargs,
            )
        except Exception as exc:
            raise RuntimeError(f"LLM structured output call failed: {exc}") from exc

        parsed = response.choices[0].message.parsed
        if parsed is None:
            # 降级：尝试 JSON 解析
            content = response.choices[0].message.content or ""
            try:
                import json

                data = json.loads(content)
                return response_model.model_validate(data)
            except Exception as json_exc:
                raise RuntimeError(f"LLM returned unparseable content: {json_exc}") from json_exc

        return parsed
