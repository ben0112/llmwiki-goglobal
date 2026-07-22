"""OpenAI 兼容 LLM 客户端(语料分类流水线用)。

纯异步、单文件依赖 httpx(api 环境已有)。端点/模型/密钥三级解析:
设置存储(前端可配)> 环境变量 > 内置默认。`base_url="mock"` 走规则桩,
无端点也能自测(与标注工具包的 --mock 对齐)。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

DEFAULT_BASE_URL = "http://host.docker.internal:8000/v1"
DEFAULT_MODEL = ""
DEFAULT_TIMEOUT = 120.0

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}


def is_local_endpoint(base_url: str) -> bool:
    """本地端点(自有算力,慢但免费)与云端端点(快,计费)的判别。"""
    if base_url == "mock":
        return True
    try:
        host = urlparse(base_url).hostname or ""
    except ValueError:
        return False
    return host in _LOCAL_HOSTS or host.endswith(".local") or host.endswith(".internal")


MAX_CONCURRENCY = 32


def default_concurrency(base_url: str) -> int:
    """端点感知的 LLM 并发默认值:本地共享算力保守,云端放宽。"""
    return 2 if is_local_endpoint(base_url) else 8


@dataclass
class LLMConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout: float = DEFAULT_TIMEOUT
    concurrency: int = 0  # 同时向端点发起的请求数;0 = 端点感知默认

    @property
    def effective_concurrency(self) -> int:
        return max(1, min(self.concurrency or default_concurrency(self.base_url),
                          MAX_CONCURRENCY))

    @property
    def is_mock(self) -> bool:
        return self.base_url == "mock"


class LLMError(Exception):
    pass


# 进程级共享 HTTP 客户端:数万次分类请求逐次新建/关闭 AsyncClient 会把
# TCP+TLS 握手做上数万遍。按事件循环缓存(CLI/测试可能多次 asyncio.run,
# 旧 loop 的客户端不可复用),timeout 逐请求传入。持 loop 对象比较身份
# 而非 id() 作键 —— 强引用在手,对象 ID 不可能被复用;旧 loop 已关闭时
# 其客户端无法异步 close,连接随 loop 失效,残骸交给 GC。
_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def _get_client() -> httpx.AsyncClient:
    global _client, _client_loop
    loop = asyncio.get_running_loop()
    if _client is None or _client.is_closed or _client_loop is not loop:
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        _client_loop = loop
    return _client


async def chat_json(config: LLMConfig, system_prompt: str, user_prompt: str,
                    *, required_keys: tuple[str, ...], strict_retry: bool = True) -> dict:
    """一次对话补全并从回复中提取扁平 JSON;格式漂移时带提示重试一次。"""
    obj = await _chat_once(config, system_prompt, user_prompt, required_keys)
    if obj is None and strict_retry:
        obj = await _chat_once(
            config, system_prompt,
            user_prompt + "\n注意:上次格式错误。只输出一个合法扁平JSON,禁止其他字符。",
            required_keys,
        )
    if obj is None:
        raise LLMError("响应中未找到合法JSON")
    return obj


async def _complete_raw(config: LLMConfig, system_prompt: str, user_prompt: str,
                        *, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        # 本地推理栈(MLX/vLLM 等)支持时关闭思考模式,省时省 token
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        resp = await _get_client().post(
            config.base_url.rstrip("/") + "/chat/completions",
            json=payload, headers=headers, timeout=config.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as e:
        raise LLMError(f"端点请求失败: {e}") from e
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"响应结构异常: {e}") from e


async def _chat_once(config: LLMConfig, system_prompt: str, user_prompt: str,
                     required_keys: tuple[str, ...]) -> dict | None:
    content = await _complete_raw(
        config, system_prompt, user_prompt, max_tokens=1024, temperature=0.1,
    )
    return extract_flat_json(content, required_keys)


async def chat_text(config: LLMConfig, system_prompt: str, user_prompt: str,
                    *, max_tokens: int = 8192, temperature: float = 0.2) -> str:
    """一次对话补全,返回纯文本(维基页面重写等长文本任务用)。"""
    content = await _complete_raw(
        config, system_prompt, user_prompt,
        max_tokens=max_tokens, temperature=temperature,
    )
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    # 剥掉模型习惯性包裹的整段代码栅栏
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)
        content = content.strip()
    if not content:
        raise LLMError("响应为空")
    return content


def extract_flat_json(content: str, required_keys: tuple[str, ...]) -> dict | None:
    """从模型输出中提取包含任一必需键的扁平 JSON(容忍思考标签/尾逗号)。"""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", content, flags=re.S))):
        try:
            cand = json.loads(m.group(0))
        except ValueError:
            continue
        if any(k in cand for k in required_keys):
            return cand
    m = re.search(r"\{.*\}", content, flags=re.S)
    if m:
        for s in (m.group(0), re.sub(r",\s*([}\]])", r"\1", m.group(0))):
            try:
                cand = json.loads(s)
            except ValueError:
                continue
            if any(k in cand for k in required_keys):
                return cand
    return None


async def probe(config: LLMConfig) -> dict:
    """测试连接:发一次最小补全,返回可达性与时延(设置页"测试连接"按钮用)。"""
    import time
    if config.is_mock:
        return {"ok": True, "latency_ms": 0, "detail": "mock 模式(规则桩,无需端点)"}
    t0 = time.monotonic()
    try:
        obj = await chat_json(
            config, "只输出 {\"ok\": true}", "输出JSON。",
            required_keys=("ok",), strict_retry=False,
        )
        latency = int((time.monotonic() - t0) * 1000)
        return {"ok": bool(obj), "latency_ms": latency, "detail": f"模型 {config.model or '(默认)'} 可达"}
    except LLMError as e:
        return {"ok": False, "latency_ms": int((time.monotonic() - t0) * 1000), "detail": str(e)}
