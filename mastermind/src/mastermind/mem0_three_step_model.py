# -*- coding: utf-8 -*-


from __future__ import annotations
import atexit
import os
import re
import time
from uuid import uuid4
from typing import List, Dict, Optional, Tuple

try:
    import msvcrt  # noqa: F401
except Exception:
    pass

import requests  # 用于直接访问 Qdrant REST API
from openai import OpenAI
from mem0 import Memory
from mastermind.models import LanguageModel

ChatHistory = List[Dict[str, str]]


class Mem0ThreeStepModel(LanguageModel):
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        user_id: str = "mm_user_01",
        recall_k: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 512,
        openai_api_key: Optional[str] = None,
        qdrant_url: str = "http://localhost:6333",
        base_collection_name: str = "mem0_mastermind",
        debug_print_search: bool = True,
    ):
        self.model_name = model_name
        self.user_id = user_id
        self.agent_id = "mastermind-assistant"
        self.run_id = f"game-{uuid4().hex[:8]}"
        self.recall_k = recall_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.debug_print_search = debug_print_search

        # 去重：已写入的 (guess, hint) 对
        self._seen_pairs: set[Tuple[Optional[Tuple[str, ...]], Optional[str]]] = set()

        # 动态解析 allowed colors
        self.allowed_colors: Optional[List[str]] = None
        # 首轮输入是否已写
        self._seed_written: bool = False

        # OpenAI
        self.OPENAI_API_KEY = (
            openai_api_key
            or os.getenv("OPENAI_API_KEY")
        )
        self.openai = OpenAI(api_key=self.OPENAI_API_KEY)

        # Qdrant 基本信息
        self.qdrant_url = qdrant_url.rstrip("/")
        self.collection_name = base_collection_name
        self.embedding_dims = 1536
        self.distance = "Cosine"

        # === 启动即自动确保集合就绪（不存在则创建；不匹配则新建一个干净集合名） ===
        self.collection_name = self._ensure_collection(
            url=self.qdrant_url,
            collection_name=self.collection_name,
            expect_dims=self.embedding_dims,
            expect_distance=self.distance,
        )

        # === Mem0 配置 ===
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "url": self.qdrant_url,
                    "collection_name": self.collection_name,
                    "embedding_model_dims": self.embedding_dims,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "api_key": self.OPENAI_API_KEY,
                    "model": "text-embedding-3-small",  # 1536 维
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "api_key": self.OPENAI_API_KEY,
                    "model": self.model_name,
                },
            },
        }
        self.memory = Memory.from_config(config)
        print(f"[mem0] Using Qdrant at {self.qdrant_url}, collection='{self.collection_name}' (dims={self.embedding_dims}, distance={self.distance})")

        # 退出时优雅关闭底层 client（若存在）
        try:
            client = getattr(self.memory, "client", None) or getattr(self.memory, "_client", None)
            if client and hasattr(client, "close"):
                atexit.register(client.close)
        except Exception:
            pass

    # ---------- 自动确保集合存在且 schema 正确 ----------
    def _ensure_collection(self, url: str, collection_name: str, expect_dims: int, expect_distance: str) -> str:
        """
        1) 检查集合是否存在：
           - 不存在：直接创建
           - 存在：校验 vectors.size & distance
                - 匹配：使用现有集合
                - 不匹配：自动生成新集合名并创建（避免复用旧坏段）
        2) 返回最终可用的集合名
        """
        def _get(path: str):
            return requests.get(f"{url}{path}", timeout=5)

        def _put(path: str, json: Dict):
            return requests.put(f"{url}{path}", json=json, timeout=10)

        # 0) 等 ready（容器刚起时稍等片刻）
        try:
            r = _get("/readyz")
            if r.status_code != 200:
                time.sleep(0.8)
        except Exception:
            time.sleep(0.8)

        # 1) 读取集合配置
        try:
            resp = _get(f"/collections/{collection_name}")
            if resp.status_code == 200:
                data = resp.json().get("result", {})
                vectors = (((data.get("config") or {}).get("params") or {}).get("vectors") or {})
                # vectors 既可能是 dict，也可能是多向量 map；这里只处理单向量的 dict 形态
                size = vectors.get("size")
                dist = vectors.get("distance")
                if size == expect_dims and str(dist).lower() == expect_distance.lower():
                    print(f"[mem0] ℹ️ collection '{collection_name}' exists and matches schema ({size}, {dist}).")
                    return collection_name
                else:
                    # 不匹配：创建一个新集合名，避免和旧段混用
                    new_name = f"{collection_name}_{time.strftime('%Y%m%d_%H%M%S')}"
                    print(f"[mem0] collection '{collection_name}' schema mismatch (got: size={size}, distance={dist}; expect: {expect_dims},{expect_distance}).")
                    print(f"[mem0] ➜ Will create a new clean collection: '{new_name}'")
                    create_body = {"vectors": {"size": expect_dims, "distance": expect_distance}}
                    resp2 = _put(f"/collections/{new_name}", json=create_body)
                    resp2.raise_for_status()
                    print(f"[mem0] created collection '{new_name}'")
                    return new_name
            elif resp.status_code == 404:
                # 不存在则创建
                create_body = {"vectors": {"size": expect_dims, "distance": expect_distance}}
                resp2 = _put(f"/collections/{collection_name}", json=create_body)
                resp2.raise_for_status()
                print(f"[mem0] created collection '{collection_name}'")
                return collection_name
            else:
                # 其它返回码：保守起见也创建一个新名
                new_name = f"{collection_name}_{time.strftime('%Y%m%d_%H%M%S')}"
                create_body = {"vectors": {"size": expect_dims, "distance": expect_distance}}
                resp2 = _put(f"/collections/{new_name}", json=create_body)
                resp2.raise_for_status()
                print(f"[mem0] created collection '{new_name}' (fallback, prev status={resp.status_code})")
                return new_name
        except Exception as e:
            # 网络/权限等异常：再试一次以新名创建
            new_name = f"{collection_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            try:
                create_body = {"vectors": {"size": expect_dims, "distance": expect_distance}}
                resp2 = requests.put(f"{url}/collections/{new_name}", json=create_body, timeout=10)
                resp2.raise_for_status()
                print(f"[mem0] created collection '{new_name}' (exception path: {e})")
                return new_name
            except Exception as e2:
                raise RuntimeError(f"[mem0] failed to ensure collection: {e}; then {e2}")

    # ---------- 工具函数 ----------
    @staticmethod
    def _extract_last_text(chat_history: ChatHistory) -> Dict[str, str]:
        last_user_text, last_assistant_text = "", ""
        for turn in reversed(chat_history):
            role = turn.get("role")
            if not last_assistant_text and role == "assistant":
                last_assistant_text = (turn.get("content") or "").strip()
            if not last_user_text and role == "user":
                last_user_text = (turn.get("content") or "").strip()
            if last_user_text and last_assistant_text:
                break
        return {"user": last_user_text, "assistant": last_assistant_text}

    @staticmethod
    def _extract_hint(hay: str) -> Optional[str]:
        if not hay:
            return None
        if "<number>" in hay:
            return None
        m = re.search(r"Feedback:\s*.*?(\d+).*?(\d+).*?(?:\n|$)", hay, re.I)
        return m.group(0).strip() if m else None

    @staticmethod
    def _extract_guess(hay: str) -> Optional[List[str]]:
        if not hay:
            return None
        m = re.search(r"FINAL GUESS:\s*\[([^\]]+)\]", hay, re.I)
        if not m:
            return None
        return [s.strip().strip("'\"") for s in m.group(1).split(",")]

    @staticmethod
    def _extract_allowed_colors(hay: str) -> Optional[List[str]]:
        if not hay:
            return None
        m = re.search(r"The following colors are allowed:\s*\[([^\]]+)\]", hay, re.I)
        if not m:
            return None
        raw = m.group(1)
        cols = [s.strip().strip("'\"") for s in raw.split(",")]
        cols = [c.lower() for c in cols if c.strip()]
        return cols or None

    # ---------- 写库“安全阀” ----------
    def _embedding_ok(self, text: str) -> bool:
        try:
            if not text or not text.strip():
                return False
            emb = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=text.strip()[:8000],
            )
            vec = (emb.data[0].embedding if emb and emb.data else None)
            ok = bool(vec) and (len(vec) == self.embedding_dims)
            if not ok:
                print("[mem0.guard] embedding length invalid -> skip write")
            return ok
        except Exception as e:
            print(f"[mem0.guard] embedding error -> skip write: {e}")
            return False

    def _safe_add(self, text: str, *, metadata: Dict, infer: bool) -> Optional[Dict]:
        if not self._embedding_ok(text):
            return None
        try:
            return self.memory.add(
                text,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.run_id,
                metadata=metadata,
                infer=infer,
            )
        except Exception as e:
            print(f"[mem0.add] error (safe-add): {e}")
            return None

    # ---------- 主流程 ----------
    def __call__(self, chat_history: ChatHistory) -> ChatHistory:
        texts = self._extract_last_text(chat_history)
        last_user_text = texts["user"]
        last_assistant_text = texts["assistant"]

        if self.allowed_colors is None:
            self.allowed_colors = self._extract_allowed_colors(last_user_text)
        if not self.allowed_colors:
            self.allowed_colors = ['purple', 'brown', 'blue', 'white', 'yellow', 'black']

        prev_hint = self._extract_hint(last_user_text)
        prev_guess = self._extract_guess(last_assistant_text)

        # 首轮规则 seed（安全阀保护）
        if not self._seed_written:
            has_rule = bool(self._extract_allowed_colors(last_user_text)) or ("Mastermind" in (last_user_text or ""))
            if has_rule and last_user_text:
                meta = {"game": self.run_id, "type": "fact"}
                res_seed = self._safe_add(last_user_text, metadata=meta, infer=True)
                print(f"[mem0.add] seed(fact,infer=True) saved: {res_seed if res_seed else {'results': []}}")
                need_fallback = (not res_seed) or (isinstance(res_seed, dict) and not res_seed.get("results"))
                if need_fallback:
                    res_seed2 = self._safe_add(last_user_text, metadata=meta, infer=False)
                    print(f"[mem0.add] seed(fact,infer=False,fallback) saved: {res_seed2 if res_seed2 else {'results': []}}")
                self._seed_written = True

        # 写入上一轮的结构化 fact（去重 + 安全阀）
        pair_key = (tuple(prev_guess) if prev_guess else None, prev_hint if prev_hint else None)
        if prev_hint or prev_guess:
            if pair_key not in self._seen_pairs:
                self._seen_pairs.add(pair_key)
                fact_parts = []
                if prev_guess:
                    fact_parts.append(f"guess: {prev_guess}")
                if prev_hint:
                    fact_parts.append(f"hint: {prev_hint}")
                fact_text = "[Mastermind Memory] " + " | ".join(fact_parts)
                meta = {"game": self.run_id, "type": "fact"}
                if prev_guess: meta["guess"] = prev_guess
                if prev_hint: meta["hint"] = prev_hint
                res_prev = self._safe_add(fact_text, metadata=meta, infer=False)
                print(f"[mem0.add] fact(prev) saved: {res_prev if res_prev else {'results': []}}")

        # ——检索（发生在生成之前）——
        if prev_guess or prev_hint:
            query = f"Mastermind memory for next guess | prev_guess={prev_guess} | feedback={prev_hint}"
        else:
            query = last_user_text or "Mastermind next guess"

        try:
            res = self.memory.search(
                query=query,
                user_id=self.user_id,
                agent_id=self.agent_id,
                run_id=self.run_id,
                limit=self.recall_k,
                filters={"game": self.run_id, "type": "fact"},
            ) or {}
            hits = []
            for item in (res.get("results") or []):
                mem_text = item.get("memory") or item.get("text")
                if mem_text:
                    hits.append(str(mem_text))
        except Exception as e:
            print(f"[mem0.search] error: {e}")
            hits = []

        if self.debug_print_search:
            print(f"[mem0.search] query = {query!r}")
            print(f"[mem0.search] hits  = {len(hits)}")
            for i, h in enumerate(hits[: self.recall_k], 1):
                print(f"  [{i}] {h}")

        # 构造 prompts
        messages: ChatHistory = []
        allowed_str = "[" + ",".join(f"'{c}'" for c in self.allowed_colors) + "]"
        base_rules = (
            "Rules:\n"
            f"- Allowed colors only: {allowed_str}.\n"
            "- Output strictly one line: FINAL GUESS:[c1, c2, c3, c4]\n"
            "- Do not repeat an identical guess from history."
        )
        messages.append({"role": "system", "content": base_rules})

        guess_hits = [h for h in hits if "[Mastermind Memory] guess:" in h]
        fallback_hits = hits if not guess_hits else []

        if guess_hits:
            messages.append({
                "role": "system",
                "content": "Use the following memory facts to improve your next guess:\n"
                           + "\n".join(f"- {h}" for h in guess_hits[: self.recall_k])
            })
        elif fallback_hits:
            messages.append({
                "role": "system",
                "content": "Context:\n" + "\n".join(f"- {h}" for h in fallback_hits[: self.recall_k])
            })

        messages.extend(chat_history)

        resp = self.openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        assistant_text = resp.choices[0].message.content or ""
        chat_history.append({"role": "assistant", "content": assistant_text})

        return chat_history

    def get_model_info(self) -> str:
        return f"Mem0 + OpenAI (3-step minimal, facts only): {self.model_name}"

    def shutdown(self):
        try:
            client = getattr(self.memory, "client", None) or getattr(self.memory, "_client", None)
            if client and hasattr(client, "close"):
                client.close()
        except Exception:
            pass
