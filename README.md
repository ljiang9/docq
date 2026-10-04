# docq

在本地 Markdown 文档上提问的极简 RAG。没有 hype 的版本：**不做 embedding、不建向量库**——把 `.md` 按标题切块，用手写的 BM25-ish 关键词打分召回相关块，再让 LLM 只根据这些块回答并标注来源。

## 诚实声明

- 检索 = 关键词匹配（tf-idf/BM25 手写实现），**不懂语义**。问"怎么退款"能命中"退费政策"，但问"钱怎么回来"可能就找不到了——这是关键词检索的固有局限。
- 回答质量取决于模型是否遵守"只根据资料回答 + 标注引用"的指令，`--dry-run` 可以先看召回质量。
- 适合：个人知识库、产品文档 FAQ、几十到几百个 Markdown 文件的场景。不适合：需要语义理解、海量文档、图片表格为主的文档。

## 安装

零依赖，Python 3.10+：

```bash
git clone https://github.com/ljiang9/docq
cd docq
python3 -m docq --help
```

## 快速开始

```bash
# 1. 建索引（walk .md，按标题切块 ~500 字，存为 .docq-index.json）
python3 -m docq index ./examples

# 2. 看索引统计
python3 -m docq stats ./examples

# 3. 先 dry-run 看召回质量（不花钱）
python3 -m docq ask "退费政策是什么？" --dir ./examples --dry-run

# 4. 真问（需要 OPENAI_API_KEY，兼容任何 OpenAI-compatible 接口）
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，第三方接口改这里
python3 -m docq ask "退费政策是什么？" --dir ./examples
```

输出示例：

```
Pro 与团队版均支持 7 天无理由退款……

—— 来源 ——
[pricing.md#定价 / 退费政策]
[faq.md#常见问题 / 账号与数据]
```

## 命令

| 命令 | 说明 |
|------|------|
| `docq index [dir]` | 建索引；无变化时按 mtime 自动跳过，`--rebuild` 强制重建 |
| `docq stats [dir]` | 每个文件的块数统计 |
| `docq ask "问题" [--dir] [--n 4]` | 召回 top-N 块，调 LLM 作答并列来源 |
| `docq ask ... --dry-run` | 只打印召回块 + 发给模型的 prompt，不调 API |

环境变量：`OPENAI_API_KEY`、`OPENAI_BASE_URL`（默认 `https://api.openai.com/v1`）、`OPENAI_MODEL`（默认 `gpt-4o-mini`），也可用 `--api-key/--base-url/--model` 覆盖。

## 工作原理

1. **切块**：按 Markdown 标题切分，标题路径（如 `定价 / 退费政策`）保留为引用；超长段落按 500 字窗口、50 字重叠再切。
2. **索引**：JSON 存块文本 + 文档频率，一个文件 `./docs/.docq-index.json` 搞定。
3. **检索**：中文单字+二元组、英文按词分词，BM25 打分取 top-N。
4. **作答**：system prompt 强制"只根据 context 回答、关键事实标注 `[文件名#标题]`"，答完打印来源列表。召回为空时直接提示，不调 API。

## License

MIT
