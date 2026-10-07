"""跨 Agent 治理共享（团队记忆域）。

把一个 agent 的 DEC/LES/STA 打包成可迁移的记忆包（pack），经治理门
导出、经完整性与策略校验导入到另一个 agent——团队共享经验，但不共享
私有上下文。

治理三铁律：
1. **分享前必须溯源可验证**（require_chain_ok）：drill_down 不通过的
   记录不许出门——无法溯源的"经验"在团队里是污染源；
2. **导入必验完整性**（hash 校验）：包被篡改即拒收，零信任搬运；
3. **策略裁决字段**（redact_keys / allow_types / min_confidence）：
   哪些字段不出门、哪些类型不进团队、低置信度不传播。

审计：每次导出/导入写 share_log（方向、计数、包哈希、时间），团队
记忆的流动留痕。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict

from .memory import DLSMemory, DLSRecord, DLSError

__all__ = ["SharePolicy", "export_pack", "import_pack", "verify_pack"]

FORMAT = "mnemos-ark-pack/1"


@dataclass
class SharePolicy:
    allow_types: tuple = ("decision", "lesson", "status")
    min_confidence: float = 0.5
    require_chain_ok: bool = True
    redact_keys: tuple = ("provenance",)
    allow_external_refs: bool = True
    max_records: int = 1000


def _canonical(records: list[dict], links: list[dict]) -> str:
    payload = json.dumps({"records": records, "links": links},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_pack(pack: dict) -> dict:
    """校验记忆包完整性（哈希）与格式。"""
    if not isinstance(pack, dict) or pack.get("format") != FORMAT:
        return {"ok": False, "reason": f"格式不支持: {pack.get('format') if isinstance(pack, dict) else type(pack).__name__}"}
    digest = _canonical(pack.get("records", []), pack.get("links", []))
    if digest != pack.get("hash"):
        return {"ok": False, "reason": "哈希不符：包已被篡改或损坏"}
    return {"ok": True, "hash": digest, "records": len(pack.get("records", []))}


def _audit(dls: DLSMemory, direction: str, project: str, count: int,
           bundle_hash: str, note: str = "") -> None:
    dls._db.execute(
        "CREATE TABLE IF NOT EXISTS share_log ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, direction TEXT, project TEXT,"
        " count INTEGER, bundle_hash TEXT, note TEXT, created_at REAL)")
    with dls._db:
        dls._db.execute(
            "INSERT INTO share_log (direction, project, count, bundle_hash,"
            " note, created_at) VALUES (?,?,?,?,?,?)",
            (direction, project, count, bundle_hash, note, time.time()))


def export_pack(dls: DLSMemory, project: str, policy: SharePolicy | None = None,
                include_links: bool = True) -> dict:
    """导出项目记忆包。治理门逐条裁决，被拒记录附理由（透明不静默）。"""
    policy = policy or SharePolicy()
    records: list[dict] = []
    rejected: list[dict] = []
    kept_ids: set[str] = set()

    for rec in dls.list_records(project=project):
        if rec.type not in policy.allow_types:
            rejected.append({"id": rec.id, "reason": f"类型不允许: {rec.type}"})
            continue
        if rec.confidence < policy.min_confidence:
            rejected.append({"id": rec.id, "reason": f"置信度 {rec.confidence} < {policy.min_confidence}"})
            continue
        if policy.require_chain_ok and not dls.drill_down(rec.id)["chain_ok"]:
            rejected.append({"id": rec.id, "reason": "下钻不变量未通过（溯源链断裂）"})
            continue
        data = asdict(rec)
        for key in policy.redact_keys:
            data[key] = ""
        records.append(data)
        kept_ids.add(rec.id)
        if len(records) >= policy.max_records:
            break

    links = []
    if include_links and kept_ids:
        placeholders = ",".join("?" * len(kept_ids))
        rows = dls._db.execute(
            f"SELECT src, dst, relation, note, created_at FROM links "
            f"WHERE src IN ({placeholders}) AND dst IN ({placeholders})",
            [*kept_ids, *kept_ids]).fetchall()
        links = [dict(r) for r in rows]

    digest = _canonical(records, links)
    pack = {
        "format": FORMAT,
        "exported_at": time.time(),
        "project": project,
        "records": records,
        "links": links,
        "rejected": rejected,
        "hash": digest,
    }
    _audit(dls, "export", project, len(records), digest,
           f"rejected={len(rejected)}")
    return pack


def import_pack(dls: DLSMemory, pack: dict, target_project: str | None = None,
                policy: SharePolicy | None = None) -> dict:
    """导入记忆包：验哈希 → 策略裁决 → 去重 → 重新编号入库 → 边重映射。"""
    policy = policy or SharePolicy()
    verdict = verify_pack(pack)
    if not verdict["ok"]:
        raise DLSError(f"记忆包校验失败: {verdict['reason']}")

    project = target_project or pack.get("project", "_imported")
    id_map: dict[str, str] = {}
    imported, skipped = 0, 0
    existing = {(r.type, r.title, r.body) for r in dls.list_records(project=project)}

    for data in pack.get("records", []):
        if data.get("type") not in policy.allow_types:
            skipped += 1
            continue
        if float(data.get("confidence", 1.0)) < policy.min_confidence:
            skipped += 1
            continue
        sources = list(data.get("sources") or [])
        if sources and not policy.allow_external_refs:
            if any(s not in {d.get("id") for d in pack.get("records", [])}
                   for s in sources):
                skipped += 1
                continue
        key = (data.get("type"), data.get("title"), data.get("body"))
        if key in existing:
            skipped += 1
            continue
        rec = dls.add(
            type=data["type"], title=data["title"], body=data.get("body", ""),
            project=project, domain=data.get("domain", "general"),
            payload=data.get("payload") or {},
            confidence=float(data.get("confidence", 1.0)),
            tags=data.get("tags") or [], provenance="imported",
            sources=sources, verify=False)
        id_map[data["id"]] = rec.id
        existing.add(key)
        imported += 1

    for edge in pack.get("links", []):
        src, dst = id_map.get(edge.get("src")), id_map.get(edge.get("dst"))
        if src and dst and src != dst:
            try:
                dls.link(src, dst, edge.get("relation", "relates"),
                         note=edge.get("note", ""))
            except DLSError:
                skipped += 1

    _audit(dls, "import", project, imported, pack.get("hash", ""),
           f"skipped={skipped}")
    return {"imported": imported, "skipped": skipped,
            "project": project, "id_map": id_map,
            "hash_ok": True}
