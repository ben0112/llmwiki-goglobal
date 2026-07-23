"""OpenAI 兼容 LLM 客户端(语料分类流水线用)。

纯异步、单文件依赖 httpx(api 环境已有)。端点/模型/密钥三级解析:
设置存储(前端可配)> 环境变量 > 内置默认。`base_url="mock"` 走规则桩,
无端点也能自测(与标注工具包的 --mock 对齐)。
"""

from __future__ import annotations

import asyncio
import json
import os
import random
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


# 并发上限:默认 256,可经 CORPUS_LLM_MAX_CONCURRENCY 放宽(封顶 1024)
# 供大配额端点实验。再往上通常先撞端点限流(429)或 vLLM 排队,拿不到
# 收益;并发只压缩耗时,token 花费不变。
MAX_CONCURRENCY = max(32, min(1024, int(
    os.environ.get("CORPUS_LLM_MAX_CONCURRENCY", "") or 256)))


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
    enable_thinking: bool = False  # 思考模式:更审慎但更慢、更耗 token

    @property
    def effective_concurrency(self) -> int:
        return max(1, min(self.concurrency or default_concurrency(self.base_url),
                          MAX_CONCURRENCY))

    @property
    def is_mock(self) -> bool:
        return self.base_url == "mock"


class LLMError(Exception):
    pass


# 大并发下的瞬时故障自愈:429(限流)/5xx(过载)与连接类瞬断在请求层
# 指数退避重试,不上抛为语料失败 —— 否则并发风暴会把好语料成片打进
# 失败态(满 3 次即不再入选队列)。
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_MAX = 5   # 全抖动 1/2/4/8/16s 级,最长累计约 45s


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """服务端给了 Retry-After 就照办(封顶 60s),否则指数退避 + 全抖动。"""
    if retry_after:
        try:
            return min(60.0, max(0.5, float(retry_after)))
        except ValueError:
            pass
    return min(30.0, float(2 ** attempt)) * (0.5 + random.random())


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
        # 连接池随并发上限走:64 硬顶会让高并发在客户端内部排队,
        # 空跑不出配置的并发数
        _client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=MAX_CONCURRENCY + 32,
                                max_keepalive_connections=min(64, MAX_CONCURRENCY)),
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
        # 思考模式开关(设置页可配,默认关):经 chat_template_kwargs 对
        # 本地推理栈(MLX/vLLM 等)生效,不支持的端点会忽略该字段。开启
        # 后分类更审慎但更慢、更耗 token;输出中的 <think> 段由解析层剥除。
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    url = config.base_url.rstrip("/") + "/chat/completions"
    client = _get_client()

    data = None
    for attempt in range(_RETRY_MAX + 1):
        try:
            resp = await client.post(url, json=payload, headers=headers,
                                     timeout=config.timeout)
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            # 连接类瞬断(端点重启/负载均衡切换):退避重跑
            if attempt >= _RETRY_MAX:
                raise LLMError(f"端点请求失败(重试 {_RETRY_MAX} 次后放弃): {e}") from e
            await asyncio.sleep(_retry_delay(attempt, None))
            continue
        except httpx.HTTPError as e:
            # 读超时等不重试:深队列下重试只会火上浇油
            raise LLMError(f"端点请求失败: {e}") from e
        if resp.status_code in _RETRY_STATUS:
            if attempt >= _RETRY_MAX:
                raise LLMError(
                    f"端点过载/限流(HTTP {resp.status_code},重试 {_RETRY_MAX} 次后放弃)")
            await asyncio.sleep(_retry_delay(attempt, resp.headers.get("retry-after")))
            continue
        try:
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise LLMError(f"端点请求失败: {e}") from e
        except ValueError as e:
            raise LLMError(f"响应非 JSON: {e}") from e
        break

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
