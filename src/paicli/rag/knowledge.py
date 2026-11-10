"""带来源版本校验的本地文档检索，组合 FTS5 BM25、稠密向量和排序融合。"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import struct
from contextlib import closing
from importlib.metadata import version
from pathlib import Path

from paicli.harness.contracts import INTERNAL_DIRS

_MODEL = "intfloat/multilingual-e5-small"
_SUFFIXES = {".md", ".txt", ".rst"}


class LocalEmbedding:
    """按需加载本地嵌入模型，并区分文档与查询的编码方式。"""

    def __init__(self, *, download=False):
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource, PoolingType
        except ImportError as exc:
            raise ValueError("Install the retrieval extra: uv sync --extra retrieval") from exc
        if _MODEL not in {m["model"] for m in TextEmbedding.list_supported_models()}:
            TextEmbedding.add_custom_model(
                model=_MODEL,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=_MODEL),
                dim=384,
                model_file="onnx/model.onnx",
                license="mit",
            )
        self.signature = f"fastembed:{version('fastembed')}:{_MODEL}:mean:prefix-v1"
        cache = os.environ.get("PAICLI_EMBED_CACHE") or str(Path.home() / ".paicli/models")
        self.model = TextEmbedding(
            _MODEL, cache_dir=cache, threads=2, cuda=False, local_files_only=not download
        )

    def encode(self, texts, *, query=False):
        prefix = "query: " if query else "passage: "
        return list(self.model.embed([prefix + text for text in texts], batch_size=32))


def _tokens(text):
    words = re.findall(r"[a-z0-9_]+", text.casefold())
    for phrase in re.findall(r"[\u3400-\u9fff]+", text):
        words.extend(phrase[i : i + 2] for i in range(len(phrase) - 1))
    return " ".join(words)


def _normalise(vector):
    values = [float(value) for value in vector]
    if not values or len(values) > 4096 or not all(math.isfinite(v) for v in values):
        raise ValueError("Invalid embedding vector")
    norm = math.sqrt(sum(value * value for value in values))
    if not norm:
        raise ValueError("Zero embedding vector")
    return [value / norm for value in values]


def _chunks(content):
    heading, buffer, start = "", [], 1
    for number, line in enumerate(content.splitlines(), 1):
        if line.startswith("#"):
            if buffer:
                yield start, number - 1, heading, "\n".join(buffer)
                buffer = []
            heading = line.lstrip("# ")[:100]
        for offset in range(0, max(1, len(line)), 240):
            piece = line[offset : offset + 240]
            if buffer and sum(len(item) + 1 for item in buffer) + len(piece) > 240:
                yield start, number if offset else number - 1, heading, "\n".join(buffer)
                buffer = []
            if not buffer:
                start = number
            buffer.append(piece)
    if buffer and any(line.strip() for line in buffer):
        yield start, number, heading, "\n".join(buffer)


class KnowledgeIndex:
    """按项目保存文档分块、版本和检索数据，为调试提供有来源的资料片段。"""

    def __init__(self, cwd, *, embedder=None, db_path=None):
        self.root = Path(cwd).resolve()
        self.scope = os.path.normcase(str(self.root))
        scope_id = hashlib.sha256(self.scope.encode()).hexdigest()
        self.path = (
            Path(db_path) if db_path else Path.home() / ".paicli/knowledge" / scope_id / "index.db"
        )
        self.embedder = embedder

    def _read(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or any(
            part.casefold() in INTERNAL_DIRS for part in path.relative_to(self.root).parts
        ):
            raise ValueError(
                "Knowledge source must stay inside the project, outside internal directories"
            )
        return path

    def index(self, sources):
        """读取并分块显式选择的资料，计算检索数据后更新项目索引。"""
        paths = set()
        for name in sources:
            path = self._read(name)
            if not path.exists():
                raise ValueError(f"Knowledge source does not exist: {name}")
            candidates = path.rglob("*") if path.is_dir() else [path]
            for item in candidates:
                if item.is_file() and item.suffix.casefold() in _SUFFIXES:
                    resolved = self._read(item)
                    paths.add(resolved)
        if not paths or len(paths) > 128:
            raise ValueError("Select between 1 and 128 Markdown/text documents")
        rows, fingerprints = [], {}
        for path in sorted(paths):
            raw = path.read_bytes()
            if len(raw) > 1024 * 1024:
                raise ValueError("A knowledge document exceeds 1 MB")
            relative = path.relative_to(self.root).as_posix()
            sha = hashlib.sha256(raw).hexdigest()
            fingerprints[relative] = sha
            for start, end, heading, text in _chunks(raw.decode("utf-8")):
                if text.strip():
                    context = f"{path.name[:60]} > {heading[:60]}\n{text}"
                    rows.append((relative, sha, start, end, heading, text, context))
                    if len(rows) > 2000:
                        raise ValueError("Knowledge index exceeds 2000 chunks")
        if not rows or len(rows) > 2000:
            raise ValueError("Knowledge index must contain 1-2000 bounded chunks")
        embedder = self.embedder or LocalEmbedding(download=True)
        vectors = [_normalise(v) for v in embedder.encode([row[-1] for row in rows])]
        if len(vectors) != len(rows) or len({len(v) for v in vectors}) != 1:
            raise ValueError("Embedding batch size or dimensions differ")
        if any(
            hashlib.sha256(self._read(p).read_bytes()).hexdigest() != sha
            for p, sha in fingerprints.items()
        ):
            raise ValueError("Knowledge documents changed during indexing")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("create table if not exists meta (key text primary key, value text)")
            existing = dict(db.execute("select key,value from meta"))
            if existing and existing.get("scope") != self.scope:
                raise ValueError("Knowledge index belongs to a different project")
            db.execute(
                "create table if not exists chunks (id integer primary key, path text, "
                "sha text, first_line int, last_line int, heading text, content text, vector blob)"
            )
            db.execute("create virtual table if not exists terms using fts5(tokens)")
            for table in ("chunks", "terms", "meta"):
                db.execute(f"delete from {table}")
            for index, (row, vector) in enumerate(zip(rows, vectors, strict=True), 1):
                db.execute(
                    "insert into chunks values (?,?,?,?,?,?,?,?)",
                    (index, *row[:6], struct.pack(f"<{len(vector)}f", *vector)),
                )
                db.execute(
                    "insert into terms(rowid,tokens) values (?,?)", (index, _tokens(row[-1]))
                )
            meta = {
                "scope": self.scope,
                "format": "1",
                "embedding": embedder.signature,
                "dimensions": str(len(vectors[0])),
            }
            db.executemany("insert into meta values (?,?)", meta.items())
        return {
            "documents": len(paths),
            "chunks": len(rows),
            "embedding": embedder.signature,
            "dimensions": len(vectors[0]),
            "index": str(self.path),
        }

    def search(self, query, *, mode="hybrid", k=3, min_similarity=0.80):
        """执行有界关键词、向量或混合检索，过滤过期来源并返回可追溯片段。"""
        if (
            mode not in {"hybrid", "lexical", "semantic"}
            or type(k) is not int
            or not 1 <= k <= 10
            or not math.isfinite(min_similarity)
            or not -1 <= min_similarity <= 1
        ):
            raise ValueError("Invalid retrieval mode or top-k")
        if not self.path.is_file() or not query.strip():
            return {"mode": mode, "hits": [], "stale_sources": [], "available": self.path.is_file()}
        with closing(sqlite3.connect(self.path)) as db:
            db.row_factory = sqlite3.Row
            meta = dict(db.execute("select key,value from meta"))
            if meta.get("scope") != self.scope or meta.get("format") != "1":
                raise ValueError("Knowledge index scope/format mismatch")
            all_rows = {row["id"]: dict(row) for row in db.execute("select * from chunks")}
            valid, stale = {}, set()
            for row in all_rows.values():
                path = self._read(row["path"])
                if row["path"] not in valid and row["path"] not in stale:
                    if (
                        path.is_file()
                        and hashlib.sha256(path.read_bytes()).hexdigest() == row["sha"]
                    ):
                        valid[row["path"]] = True
                    else:
                        stale.add(row["path"])
            rows = {i: row for i, row in all_rows.items() if row["path"] in valid}
            terms = list(dict.fromkeys(_tokens(query[:2000]).split()))[:64]
            lexical = []
            if terms and mode != "semantic":
                expression = " OR ".join(f'"{term}"' for term in terms)
                lexical = [
                    row[0]
                    for row in db.execute(
                        "select rowid from terms where terms match ? order by bm25(terms),rowid",
                        (expression,),
                    )
                    if row[0] in rows
                ][:20]
        dense, similarities = [], {}
        if rows and mode != "lexical":
            embedder = self.embedder or LocalEmbedding()
            if embedder.signature != meta["embedding"]:
                raise ValueError("Embedding contract changed; rebuild the knowledge index")
            vector = _normalise(embedder.encode([query[:2000]], query=True)[0])
            if len(vector) != int(meta["dimensions"]):
                raise ValueError("Embedding query dimension mismatch")
            for identifier, row in rows.items():
                values = struct.unpack(f"<{len(vector)}f", row["vector"])
                similarities[identifier] = sum(a * b for a, b in zip(vector, values, strict=True))
            dense = sorted(
                (i for i in rows if similarities[i] >= min_similarity),
                key=lambda i: (-similarities[i], i),
            )[:20]
        scores = {}
        for ranking in (lexical, dense):
            for rank, identifier in enumerate(ranking, 1):
                scores[identifier] = scores.get(identifier, 0) + 1 / (60 + rank)
        hits, counts = [], {}
        for identifier in sorted(scores, key=lambda i: (-scores[i], -similarities.get(i, 0), i)):
            row = rows[identifier]
            if counts.get(row["path"], 0) >= 2:
                continue
            counts[row["path"]] = counts.get(row["path"], 0) + 1
            hits.append(
                {key: value for key, value in row.items() if key not in {"id", "vector"}}
                | {"score": scores[identifier], "similarity": similarities.get(identifier)}
            )
            if len(hits) == k:
                break
        return {
            "mode": mode,
            "available": True,
            "hits": hits,
            "stale_sources": sorted(stale),
            "embedding": meta["embedding"] if mode != "lexical" else None,
        }
