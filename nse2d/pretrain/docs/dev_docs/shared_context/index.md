# shared_context Index

`shared_context/` 允许容纳多种共享记录，但必须满足下面至少一条：

1. 被两个及以上 `Qxx` 复用
2. 脱离具体 `Qxx` 之后仍然成立
3. 属于 codebase、资源、工具、数据、baseline 的稳定背景

## 共享文件索引

| 文件 | 类型 | 适用范围 | 来源QID | 是否稳定共识 | 最近更新时间 |
|------|------|----------|---------|--------------|--------------|
| `common_context.md` | 项目背景与稳定约束 | 全项目 | init / Q01 | 是 | 2026-03-11 |
| `codebase_resource_map.md` | 代码与资源地图 | 全项目 | init / Q01 | 是 | 2026-03-11 |
| `evaluation_protocol.md` | 评估协议与 baseline 口径 | 全项目 | Q01 | 基本是，含少量 caveat | 2026-03-11 |

