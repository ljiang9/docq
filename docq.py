#!/usr/bin/env python3
"""docq — 在本地 Markdown 文档上提问的极简 RAG。

诚实声明：没有 embedding，没有向量数据库。检索 = 手写的
BM25-ish 关键词打分（tf-idf 族）。语义理解交给 LLM，证据来自你的文档。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

VERSION = "0.1.0"
CHUNK_SIZE = 500      # 每块目标字符数
CHUNK_OVERLAP = 50    # 长段落切分时的重叠字符数
INDEX_NAME = ".docq-index.json"


class DocqError(Exception):
    pass


# ---------------------------------------------------------------- 分词
_ALNUM = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文数字按词切，中文按相邻二元组。

    注意：中文单字刻意不参与索引——单字（如"的/与/子"）在各文档中
    无处不在，只会带来噪声；二元组才是中文关键词检索的有效单位。
    代价：单个汉字组成的查询词无法被召回（实践中极少见）。
    """
    s = text.lower()
    toks = _ALNUM.findall(s)
    for run in _CJK_RUN.findall(s):
        toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    return toks


# ---------------------------------------------------------------- 切块
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """按标题把 Markdown 切成 (标题路径, 正文) 段落，长段再按窗口切分。

    返回 [(heading_path, chunk_text), ...]，heading_path 如 "定价 / 退费政策"。
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []      # (level, title)
    buf: list[str] = []

    def flush():
        if buf:
            path = " / ".join(t for _, t in stack) if stack else ""
            sections.append((path, "".join(buf).strip()))
            buf.clear()

    for line in text.splitlines(keepends=True):
        m = _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)
    flush()

    chunks: list[tuple[str, str]] = []
    for path, body in sections:
        body = body.strip()
        if not body:
            continue
        if len(body) <= CHUNK_SIZE:
            chunks.append((path, body))
        else:
            start = 0
            while start < len(body):
                end = min(start + CHUNK_SIZE, len(body))
                chunks.append((path, body[start:end].strip()))
                if end == len(body):
                    break
                start = end - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------- 建索引
def build_index(docs_dir: str, rebuild: bool = False) -> dict:
    docs_dir = os.path.abspath(docs_dir)
    index_path = os.path.join(docs_dir, INDEX_NAME)

    md_files: dict[str, float] = {}
    for root, dirs, files in os.walk(docs_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.endswith(".md"):
                fp = os.path.join(root, fn)
                md_files[os.path.relpath(fp, docs_dir)] = os.path.getmtime(fp)

    if not md_files:
        raise DocqError(f"目录 {docs_dir} 下没有找到 .md 文件")

    old = None
    if os.path.exists(index_path) and not rebuild:
        try:
            with open(index_path, encoding="utf-8") as f:
                old = json.load(f)
        except (json.JSONDecodeError, OSError):
            old = None
    if old and old.get("files") == {k: {"mtime": v, "chunks": old["files"][k]["chunks"]}
                                    for k, v in md_files.items()
                                    if k in old.get("files", {})}:
        return {"skipped": True, "path": index_path, "files": len(md_files)}

    chunks: list[dict] = []
    files_meta: dict[str, dict] = {}
    for rel, mtime in sorted(md_files.items()):
        with open(os.path.join(docs_dir, rel), encoding="utf-8") as f:
            text = f.read()
        parts = chunk_markdown(text)
        files_meta[rel] = {"mtime": mtime, "chunks": len(parts)}
        for path, body in parts:
            chunks.append({"id": len(chunks), "file": rel,
                           "heading": path or rel, "text": body})

    # 文档频率（供 BM25 用）
    df: dict[str, int] = {}
    lens: list[int] = []
    for ch in chunks:
        toks = tokenize(ch["heading"] + "\n" + ch["text"])
        lens.append(len(toks))
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    index = {
        "version": 1,
        "files": files_meta,
        "chunks": chunks,
        "docfreq": df,
        "n": len(chunks),
        "avglen": sum(lens) / len(lens) if lens else 0,
        "lens": lens,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    return {"skipped": False, "path": index_path,
            "files": len(md_files), "chunks": len(chunks)}


def load_index(docs_dir: str) -> dict:
    index_path = os.path.join(os.path.abspath(docs_dir), INDEX_NAME)
    if not os.path.exists(index_path):
        raise DocqError(f"找不到索引 {index_path}，请先运行 `docq index {docs_dir}`")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 检索（BM25-ish，手写）
def bm25_scores(query: str, index: dict) -> list[tuple[float, dict]]:
    q_terms = tokenize(query)
    if not q_terms:
        return []
    N = index["n"]
    avglen = index["avglen"] or 1
    df = index["docfreq"]
    k1, b = 1.2, 0.75
    scored = []
    for i, ch in enumerate(index["chunks"]):
        toks = tokenize(ch["heading"] + "\n" + ch["text"])
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = index["lens"][i] or 1
        score = 0.0
        for t in set(q_terms):
            if t not in df or t not in tf:
                continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            f = tf[t]
            score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avglen))
        if score > 0:
            scored.append((score, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ---------------------------------------------------------------- 问答
SYSTEM_PROMPT = """你是一个只基于所给资料回答问题的助手。严格遵守：
1. 只能使用 <context> 中的信息回答，绝不使用资料之外的知识。
2. 资料中没有相关信息时，明确说"资料中没有相关信息"，不要编造。
3. 关键事实后面用 [文件名#标题路径] 标注来源，例如 [pricing.md#退费政策]。
4. 用中文回答，简洁直接。"""


def build_messages(question: str, hits: list[tuple[float, dict]]) -> list[dict]:
    parts = []
    for score, ch in hits:
        parts.append(f"[{ch['file']}#{ch['heading']}]\n{ch['text']}")
    context = "\n\n---\n\n".join(parts)
    user = f"<context>\n{context}\n</context>\n\n问题：{question}"
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def chat_complete(api_key: str, base_url: str, model: str,
                   messages: list[dict], timeout: int = 120) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "messages": messages,
                          "temperature": 0.2}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise DocqError(f"API 请求失败 (HTTP {e.code}): {detail}")
    except urllib.error.URLError as e:
        raise DocqError(f"网络请求失败: {e.reason}")
    except json.JSONDecodeError:
        raise DocqError("API 返回的不是合法 JSON")
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise DocqError("API 返回格式异常，缺少 choices[0].message.content")


# ---------------------------------------------------------------- CLI
def cmd_index(args) -> int:
    try:
        r = build_index(args.dir, rebuild=args.rebuild)
    except DocqError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if r["skipped"]:
        print(f"索引无变化，跳过（{r['files']} 个文件，索引：{r['path']}）")
    else:
        print(f"索引完成：{r['files']} 个文件，{r['chunks']} 个块 → {r['path']}")
    return 0


def cmd_stats(args) -> int:
    try:
        index = load_index(args.dir)
    except DocqError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"文档目录：{os.path.abspath(args.dir)}")
    total = 0
    for rel in sorted(index["files"]):
        n = index["files"][rel]["chunks"]
        total += n
        print(f"  {rel}: {n} 块")
    print(f"共 {len(index['files'])} 个文件，{total} 个块（索引构建于 {index.get('built_at', '?')}）")
    return 0


def cmd_ask(args) -> int:
    try:
        index = load_index(args.dir)
    except DocqError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    hits = bm25_scores(args.question, index)[:args.n]
    if not hits:
        print("未找到相关内容，未调用 API。可以尝试换个问法，或先 `docq index` 重建索引。")
        return 0

    messages = build_messages(args.question, hits)

    if args.dry_run:
        print("=== 检索到的块（dry-run，未调用 API）===")
        for score, ch in hits:
            print(f"\n[得分 {score:.2f}] {ch['file']}#{ch['heading']}")
            preview = ch["text"][:200].replace("\n", " ")
            print(f"  {preview}{'...' if len(ch['text']) > 200 else ''}")
        print("\n=== 发给模型的 prompt ===")
        for m in messages:
            print(f"\n--- {m['role']} ---\n{m['content']}")
        return 0

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: 未找到 API key。请设置 OPENAI_API_KEY 环境变量，或用 --api-key 传入。",
              file=sys.stderr)
        return 1
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    try:
        answer = chat_complete(api_key, base_url, model, messages)
    except DocqError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(answer.strip())
    print("\n—— 来源 ——")
    for _, ch in hits:
        print(f"[{ch['file']}#{ch['heading']}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docq",
                                description="在本地 Markdown 文档上提问的极简 RAG（关键词检索，无 embedding）")
    p.add_argument("--version", action="version", version=f"docq {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("index", help="为文档目录建立索引")
    pi.add_argument("dir", nargs="?", default="./docs", help="文档目录（默认 ./docs）")
    pi.add_argument("--rebuild", action="store_true", help="强制重建索引")
    pi.set_defaults(func=cmd_index)

    pa = sub.add_parser("ask", help="基于文档提问")
    pa.add_argument("question", help="问题")
    pa.add_argument("--dir", default="./docs", help="文档目录（默认 ./docs）")
    pa.add_argument("--n", type=int, default=4, help="取回的块数（默认 4）")
    pa.add_argument("--dry-run", action="store_true", help="只展示检索块和 prompt，不调 API")
    pa.add_argument("--api-key", default=None, help="API key（默认读 OPENAI_API_KEY）")
    pa.add_argument("--base-url", default=None, help="API base URL（默认读 OPENAI_BASE_URL）")
    pa.add_argument("--model", default=None, help="模型（默认读 OPENAI_MODEL）")
    pa.set_defaults(func=cmd_ask)

    ps = sub.add_parser("stats", help="查看索引统计")
    ps.add_argument("dir", nargs="?", default="./docs", help="文档目录（默认 ./docs）")
    ps.set_defaults(func=cmd_stats)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
