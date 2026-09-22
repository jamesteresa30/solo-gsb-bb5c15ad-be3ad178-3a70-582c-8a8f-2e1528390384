"""SQLite 存储层（标准库 sqlite3，线程安全封装）。"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DB_PATH = os.environ.get("DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "detections.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id                    TEXT PRIMARY KEY,
    series_id             TEXT,
    created_at            TEXT NOT NULL,
    request_params        TEXT NOT NULL,
    result_json           TEXT NOT NULL,
    point_count           INTEGER NOT NULL,
    anomaly_segment_count INTEGER NOT NULL,
    anomaly_point_count   INTEGER NOT NULL,
    threshold             REAL NOT NULL,
    window_size           INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_detections_series_id ON detections(series_id);
CREATE INDEX IF NOT EXISTS idx_detections_created_at ON detections(created_at);

-- 每个异常段的处置状态（open -> acknowledged -> resolved）；
-- 新检测入库时按段预插入，历史检测由 init_db 幂等回填。
CREATE TABLE IF NOT EXISTS segment_status (
    detection_id TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',
    note          TEXT,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (detection_id, segment_index),
    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
);
"""

# 只允许 open -> acknowledged -> resolved 的单向流转
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "open": frozenset({"acknowledged"}),
    "acknowledged": frozenset({"resolved"}),
    "resolved": frozenset(),
}


class SegmentStatusConflict(Exception):
    """非法状态跳转：当前状态不允许转换为目标状态（对应 HTTP 409）。"""

    def __init__(self, current: str, target: str) -> None:
        self.current = current
        self.target = target
        super().__init__(
            f"非法状态跳转: {current} -> {target}（仅允许 open -> acknowledged -> resolved）"
        )


def init_db(db_path: str | None = None) -> None:
    """建库建表并启用 WAL；为历史检测回填缺失的异常段处置状态。"""
    db_path = db_path or DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA)
        _backfill_segment_status(conn)
        conn.commit()


@contextmanager
def get_conn(db_path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path or DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=15000")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_detection(
    detection_id: str,
    series_id: str | None,
    request_params: dict[str, Any],
    result: dict[str, Any],
    db_path: str | None = None,
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    segment_count = int(result["anomaly_segment_count"])
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO detections (
                id, series_id, created_at, request_params, result_json,
                point_count, anomaly_segment_count, anomaly_point_count,
                threshold, window_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                detection_id,
                series_id,
                created_at,
                json.dumps(request_params, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                result["point_count"],
                segment_count,
                result["anomaly_point_count"],
                result["threshold"],
                result["window_size"],
            ),
        )
        # 新检测的异常段初始一律为 open（updated_at 取检测创建时间）
        conn.executemany(
            """
            INSERT INTO segment_status (detection_id, segment_index, status, note, updated_at)
            VALUES (?, ?, 'open', NULL, ?)
            """,
            [(detection_id, idx, created_at) for idx in range(segment_count)],
        )
    return created_at


def get_detection(detection_id: str, db_path: str | None = None) -> dict[str, Any] | None:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?", (detection_id,)
        ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "series_id": row["series_id"],
        "created_at": row["created_at"],
        "request_params": json.loads(row["request_params"]),
        "result": json.loads(row["result_json"]),
    }


def get_segment_statuses(
    detection_id: str, db_path: str | None = None
) -> list[dict[str, Any]]:
    """按段序号返回某检测全部异常段的处置状态行。"""
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT segment_index, status, note, updated_at
            FROM segment_status WHERE detection_id = ?
            ORDER BY segment_index
            """,
            (detection_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_status_counts(
    detection_ids: list[str], db_path: str | None = None
) -> dict[str, dict[str, int]]:
    """批量统计多条检测的各处置状态段数：{detection_id: {open, acknowledged, resolved}}。"""
    counts = {
        did: {"open": 0, "acknowledged": 0, "resolved": 0} for did in detection_ids
    }
    if not detection_ids:
        return counts
    with get_conn(db_path) as conn:
        rows = conn.execute(
            """
            SELECT detection_id, status, COUNT(*) AS n
            FROM segment_status WHERE detection_id IN ({})
            GROUP BY detection_id, status
            """.format(",".join("?" * len(detection_ids))),
            detection_ids,
        ).fetchall()
    for row in rows:
        counts[row["detection_id"]][row["status"]] = row["n"]
    return counts


def set_segment_status(
    detection_id: str,
    segment_index: int,
    target_status: str,
    note: str | None,
    db_path: str | None = None,
) -> dict[str, Any] | None:
    """更新异常段处置状态。

    - 检测不存在或段序号越界返回 None（路由层转 404）；
    - 重复提交与当前状态相同的目标状态视为幂等：直接返回当前行，
      note / updated_at 不变；
    - 非允许的跳转抛 SegmentStatusConflict（路由层转 409）；
    - 正常跳转时把 note 置为请求值（省略即清空），updated_at 刷为当前 UTC 时间。

    另外防御性地为历史检测缺失的状态行（init_db 回填前的边缘情况）补 open 行。
    """
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT anomaly_segment_count FROM detections WHERE id = ?",
            (detection_id,),
        ).fetchone()
        if row is None:
            return None
        segment_count = int(row["anomaly_segment_count"])
        if not 0 <= segment_index < segment_count:
            return None

        status_row = conn.execute(
            "SELECT status, note, updated_at FROM segment_status WHERE detection_id = ? AND segment_index = ?",
            (detection_id, segment_index),
        ).fetchone()
        now = datetime.now(timezone.utc).isoformat()
        if status_row is None:
            # 历史检测缺行（正常情况下 init_db 已回填）：补一行 open
            conn.execute(
                """
                INSERT INTO segment_status (detection_id, segment_index, status, note, updated_at)
                VALUES (?, ?, 'open', NULL, ?)
                """,
                (detection_id, segment_index, now),
            )
            current_status = "open"
            current_note: str | None = None
        else:
            current_status = status_row["status"]
            current_note = status_row["note"]

        if target_status == current_status:
            # 幂等：重复提交同一状态不改动 note / updated_at
            return {
                "detection_id": detection_id,
                "segment_index": segment_index,
                "status": current_status,
                "note": current_note,
                "updated_at": status_row["updated_at"] if status_row is not None else now,
            }
        if target_status not in ALLOWED_TRANSITIONS[current_status]:
            raise SegmentStatusConflict(current_status, target_status)

        conn.execute(
            """
            UPDATE segment_status SET status = ?, note = ?, updated_at = ?
            WHERE detection_id = ? AND segment_index = ?
            """,
            (target_status, note, now, detection_id, segment_index),
        )
    return {
        "detection_id": detection_id,
        "segment_index": segment_index,
        "status": target_status,
        "note": note,
        "updated_at": now,
    }


def _backfill_segment_status(conn: sqlite3.Connection) -> None:
    """为历史检测补齐缺失的异常段状态行（默认 open，updated_at=检测创建时间）。"""
    rows = conn.execute(
        """
        SELECT d.id, d.created_at, d.anomaly_segment_count
        FROM detections d
        WHERE d.anomaly_segment_count > 0
          AND NOT EXISTS (
              SELECT 1 FROM segment_status s
              WHERE s.detection_id = d.id AND s.segment_index = 0
          )
        """
    ).fetchall()
    for row in rows:
        conn.executemany(
            """
            INSERT OR IGNORE INTO segment_status (detection_id, segment_index, status, note, updated_at)
            VALUES (?, ?, 'open', NULL, ?)
            """,
            [
                (row["id"], idx, row["created_at"])
                for idx in range(int(row["anomaly_segment_count"]))
            ],
        )


def list_detections(
    series_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    with get_conn(db_path) as conn:
        if series_id is not None:
            total = conn.execute(
                "SELECT COUNT(*) FROM detections WHERE series_id = ?", (series_id,)
            ).fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, series_id, created_at, point_count,
                       anomaly_segment_count, anomaly_point_count,
                       threshold, window_size
                FROM detections WHERE series_id = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (series_id, limit, offset),
            ).fetchall()
        else:
            total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
            rows = conn.execute(
                """
                SELECT id, series_id, created_at, point_count,
                       anomaly_segment_count, anomaly_point_count,
                       threshold, window_size
                FROM detections ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows], total
