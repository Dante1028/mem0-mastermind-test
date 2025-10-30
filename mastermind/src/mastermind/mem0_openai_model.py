# -*- coding: utf-8 -*-


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional
import os
import re
import uuid

from mastermind.models import LanguageModel

# 你若不需要检索，想最干净，可把下面两行注释掉，并把 use_memory=False
from mem0 import Memory
from openai import OpenAI

ChatHistory = List[Dict[str, str]]


@dataclass
class Mem0Config:
    # 基础
    chat_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512

    # 检索（可关）
    use_memory: bool = True          # ← 若你不想检索，设 False
    top_k: int = 3
    qdrant_url: str = "http://localhost:6333"
    collection: str = "mem0_mastermind"
    embedding_dims: int = 1536
    embedding_model: str = "text-embedding-3-small"

    # 识别/隔离（只用于检索过滤）
    user_id: str = "mm_user_01"
    agent_id: str = "mastermind-assistant"
    run_id: Optional[str] = None     # 不传则自动生成

    # OpenAI Key（可用环境变量）
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL", None)


class Mem0OpenAIModel(LanguageModel):
    """
    极简：__call__(chat_history)->chat_history
      - 打印上一轮反馈（从上一条 user 文本解析）
      -（可选）检索 Mem0，打印 query 和命中
      - 生成本轮回复，并打印 FINAL GUESS
    """

    def __init__(self, cfg: Optional[Mem0Config] = None):
        self.cfg = cfg or Mem0Config()
        if not self.cfg.run_id:
            self.cfg.run_id = f"run-{uuid.uuid4().hex[:8]}"

        # OpenAI 客户端
        if self.cfg.openai_base_url:
            self.client = OpenAI(api_key=self.cfg.openai_api_key, base_url=self.cfg.openai_base_url)
        else:
            self.client = OpenAI(api_key=self.cfg.openai_api_key)

        # Mem0（仅搜索）
        self.mem = None
        if self.cfg.use_memory:
            self.mem = Memory.from_config({
                "llm": {
                    "provider": "openai",
                    "config": {"model": self.cfg.chat_model, "api_key": self.cfg.openai_api_key},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": self.cfg.embedding_model, "api_key": self.cfg.openai_api_key},
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "url": self.cfg.qdrant_url,
                        "collection_name": self.cfg.collection,
                        "embedding_model_dims": self.cfg.embedding_dims,
                    },
                },
            })

    def get_model_info(self) -> str:
        return f"Mem0+OpenAI(minimal): {self.cfg.chat_model}"

    # --------- 简单解析工具 --------- #
    @staticmethod
    def _last_user_and_assistant(chat: ChatHistory) -> tuple[str, str]:
        user, assistant = "", ""
        for m in reversed(chat):
            role = m.get("role")
            if not assistant and role == "assistant":
                assistant = (m.get("content") or "").strip()
            if not user and role == "user":
                user = (m.get("content") or "").strip()
            if user and assistant:
                break
        return user, assistant

    @staticmethod
    def _extract_feedback(text: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(r"Feedback:\s*.*?(?:\n|$)", text, re.I)
        return m.group(0).strip() if m else None

    @staticmethod
    def _extract_guess(text: str) -> Optional[List[str]]:
        if not text:
            return None
        m = re.search(r"FINAL GUESS:\s*\[([^\]]+)\]", text, re.I)
        if not m:
            return None
        return [s.strip().strip("'\"") for s in m.group(1).split(",")]

    # --------- 主流程 --------- #
    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        last_user, last_assistant = self._last_user_and_assistant(chat_history)

        # 打印上一轮：反馈 & 猜测（若有）
        prev_feedback = self._extract_feedback(last_user)
        prev_guess = self._extract_guess(last_assistant)
        if prev_guess:
            print(f"[Guess Prev] {prev_guess}")
        if prev_feedback:
            print(f"[Feedback Prev] {prev_feedback}")

        # 可选：检索 Mem0（只读），把命中拼到 system
        messages: ChatHistory = []
        if self.mem is not None:
            try:
                query = last_user or "Mastermind next guess"
                res = self.mem.search(
                    query=query,
                    user_id=self.cfg.user_id,
                    agent_id=self.cfg.agent_id,
                    run_id=self.cfg.run_id,
                    limit=self.cfg.top_k,
                ) or {}
                items = res.get("results") or []
                hits = []
                for it in items:
                    # 兼容不同字段命名
                    t = it.get("memory") or it.get("text") or it.get("content")
                    if t:
                        hits.append(str(t))
                # 打印检索日志
                print(f"[mem0.search] query = {query!r}")
                print(f"[mem0.search] hits  = {len(hits)}")
                for i, h in enumerate(hits[: self.cfg.top_k], 1):
                    print(f"  [{i}] {h}")
                if hits:
                    messages.append({
                        "role": "system",
                        "content": "Context:\n" + "\n".join(f"- {h}" for h in hits[: self.cfg.top_k])
                    })
            except Exception as e:
                print(f"[mem0.search] error: {e}")

        # 正常对话
        messages.extend(chat_history)

        resp = self.client.chat.completions.create(
            model=self.cfg.chat_model,
            messages=messages,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
        )
        assistant_text = resp.choices[0].message.content or ""
        chat_history.append({"role": "assistant", "content": assistant_text})

        # 打印本轮预测
        curr_guess = self._extract_guess(assistant_text)
        print(f"[Guess Predicted] {curr_guess if curr_guess else '(parse-failed)'}")

        return chat_history
