"""SQLite 存储层。

版本化规则（核心）：
- 每次成功提交样本或目标修改（PUT/DELETE）都会令全局 epoch +1；
- epoch 变更会删除全部旧作业记录，并置位取消事件，使在途计算永久失效
  （即使把数据改回原值也不会恢复旧结果，必须重新计算）；
- 作业登记是原子的：输入齐全性检查、epoch 与输入快照在同一临界区内完成，
  三者永远对应同一份已提交的样本与目标；输入不全时不登记任何作业；
- 结果只在作业成功完成时随作业原子写入，且写入前原子核对 epoch，
  不存在“半份结果”，在途作业被取消后绝不落库；
- 读取侧（作业查询 / /result）只交付 epoch 与快照都和当前已提交输入
  逐字段一致的作业；沿用旧库时，错配记录被跳过/视为失效，
  不会挡住当前版本的有效结果；
- 进程重启后所有未完成作业标记为 interrupted，不开放任何部分结果。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    epoch INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    person_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    age_group TEXT NOT NULL,
    d INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    dimension TEXT NOT NULL,
    code TEXT NOT NULL,
    population INTEGER NOT NULL,
    PRIMARY KEY (dimension, code)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    status TEXT NOT NULL,            -- queued|running|succeeded|infeasible|failed|interrupted
    submitted_at TEXT NOT NULL,
    finished_at TEXT,
    input_snapshot TEXT NOT NULL,    -- 提交时刻完整输入快照（JSON）
    result TEXT,                     -- 仅终态写入（完整权重+依据，或冲突集，或错误）
    error TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._write_lock = threading.RLock()
        self._cancel = threading.Event()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        cur = self._conn.execute("SELECT epoch FROM meta WHERE id=1")
        row = cur.fetchone()
        if row is None:
            self._conn.execute("INSERT INTO meta(id, epoch) VALUES (1, 1)")
            self._conn.commit()
            self.epoch = 1
        else:
            self.epoch = int(row["epoch"])
        # 重启：未完成作业一律中断，不允许读到半成品
        self._conn.execute(
            "UPDATE jobs SET status='interrupted', finished_at=?, "
            "error='process restarted' WHERE status IN ('queued','running')",
            (_now(),),
        )
        self._conn.commit()

    def close(self) -> None:
        with self._write_lock:
            self._conn.close()

    # ------------------------------------------------------------------ 读取
    def get_epoch(self) -> int:
        with self._write_lock:
            return self.epoch

    def list_samples(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT person_id, region, age_group, d FROM samples ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_targets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT dimension, code, population FROM targets ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def snapshot(self) -> dict:
        with self._write_lock:
            return {
                "epoch": self.epoch,
                "samples": self.list_samples(),
                "targets": self.list_targets(),
            }

    def get_job(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def _inputs_match_locked(self, snap: dict) -> bool:
        """调用方必须持有 _write_lock。

        作业快照须与当前已提交的样本/目标逐字段一致，且快照内 epoch
        等于当前 epoch（“改回原值”后 epoch 已变，内容相同也不算一致）。
        """
        return self._snap_matches(
            snap, self.epoch, self.list_samples(), self.list_targets()
        )

    @staticmethod
    def _snap_matches(snap: dict, epoch: int,
                      samples: list[dict], targets: list[dict]) -> bool:
        return (
            snap.get("epoch") == epoch
            and snap.get("samples") == samples
            and snap.get("targets") == targets
        )

    @staticmethod
    def _parse_snapshot(row: dict) -> dict | None:
        try:
            snap = json.loads(row["input_snapshot"])
        except (TypeError, ValueError):
            return None
        return snap if isinstance(snap, dict) else None

    def get_current_job(self, job_id: str) -> dict | None:
        """仅当作业存在、且其 epoch 与快照都和当前已提交输入一致时返回该作业。

        错配版本（旧 epoch，或旧库遗留的“旧快照登记为新版本”记录）一律
        视为已永久失效，返回 None，绝不向外交付其权重。
        """
        with self._write_lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            row = dict(row)
            if row["epoch"] != self.epoch:
                return None
            snap = self._parse_snapshot(row)
            if snap is None or not self._inputs_match_locked(snap):
                return None
            return row

    def latest_consistent_succeeded_job(self) -> dict | None:
        """返回与当前已提交输入逐字段一致的最近成功作业。

        错配记录（旧 epoch，或旧库遗留的快照与版本不符的记录）直接跳过，
        不会挡住当前版本的有效结果。
        """
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE status='succeeded' AND epoch=? "
                "ORDER BY finished_at DESC LIMIT 200",
                (self.epoch,),
            ).fetchall()
            samples, targets = self.list_samples(), self.list_targets()
            for row in rows:
                row = dict(row)
                snap = self._parse_snapshot(row)
                if snap is not None and self._snap_matches(
                        snap, self.epoch, samples, targets):
                    return row
            return None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """列出当前版本的最近作业；错配记录（旧库遗留）不列出。"""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs WHERE epoch=? "
                "ORDER BY submitted_at DESC LIMIT 200",
                (self.epoch,),
            ).fetchall()
            samples, targets = self.list_samples(), self.list_targets()
            out = []
            for row in rows:
                row = dict(row)
                snap = self._parse_snapshot(row)
                if snap is None or not self._snap_matches(
                        snap, self.epoch, samples, targets):
                    continue
                out.append({
                    "job_id": row["job_id"],
                    "epoch": row["epoch"],
                    "status": row["status"],
                    "submitted_at": row["submitted_at"],
                    "finished_at": row["finished_at"],
                    "error": row["error"],
                })
                if len(out) >= limit:
                    break
            return out

    # ------------------------------------------------------------------ 写入
    def _bump_epoch_locked(self) -> int:
        """调用方必须持有 _write_lock。"""
        self.epoch += 1
        self._conn.execute("UPDATE meta SET epoch=? WHERE id=1", (self.epoch,))
        # 旧作业（含在途）永久失效
        self._conn.execute("DELETE FROM jobs")
        self._cancel.set()
        # 唤醒等待者后立即重建事件，供下一轮作业使用
        self._cancel = threading.Event()
        return self.epoch

    def replace_samples(self, samples: list[dict]) -> int:
        with self._write_lock:
            try:
                self._conn.execute("DELETE FROM samples")
                self._conn.executemany(
                    "INSERT INTO samples(person_id, region, age_group, d) "
                    "VALUES (:person_id, :region, :age_group, :d)",
                    samples,
                )
                epoch = self._bump_epoch_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return epoch

    def replace_targets(self, targets: list[dict]) -> int:
        with self._write_lock:
            try:
                self._conn.execute("DELETE FROM targets")
                self._conn.executemany(
                    "INSERT INTO targets(dimension, code, population) "
                    "VALUES (:dimension, :code, :population)",
                    targets,
                )
                epoch = self._bump_epoch_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return epoch

    # ------------------------------------------------------------------ 作业
    def current_cancel_event(self) -> threading.Event:
        with self._write_lock:
            return self._cancel

    def register_job(self) -> tuple[dict | None, str | None]:
        """原子登记作业：输入齐全性检查、epoch、输入快照在同一临界区内完成。

        登记下来的作业，其版本号与快照永远对应同一份已提交的样本与目标——
        不会在通过检查后拿到空输入，也不会把旧快照登记为新版本。

        返回 (job, None)；当前版本缺少样本或目标时返回
        (None, "samples"|"targets")，且不登记任何作业。
        """
        with self._write_lock:
            samples = self.list_samples()
            targets = self.list_targets()
            if not samples:
                return None, "samples"
            if not targets:
                return None, "targets"
            snapshot = {
                "epoch": self.epoch,
                "samples": samples,
                "targets": targets,
            }
            job_id = uuid.uuid4().hex
            self._conn.execute(
                "INSERT INTO jobs(job_id, epoch, status, submitted_at, input_snapshot) "
                "VALUES (?,?,?,?,?)",
                (job_id, self.epoch, "queued", _now(),
                 json.dumps(snapshot, ensure_ascii=False)),
            )
            self._conn.commit()
            return self.get_job(job_id), None

    def mark_running(self, job_id: str) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE jobs SET status='running' WHERE job_id=? AND status='queued'",
                (job_id,),
            )
            self._conn.commit()

    def finish_job_if_current(self, job_id: str, epoch: int, status: str,
                              result: dict | None, error: str | None) -> bool:
        """仅当库内 epoch 仍等于作业 epoch 时原子写入终态；返回是否写入。

        epoch 检查与 UPDATE 在同一临界区内完成，且 UPDATE 自带 epoch 条件：
        输入一旦变更，在途作业的结果永久作废，绝不落库。
        """
        with self._write_lock:
            if self.epoch != epoch:
                return False
            cur = self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, result=?, error=? "
                "WHERE job_id=? AND epoch=?",
                (status, _now(),
                 json.dumps(result, ensure_ascii=False) if result is not None else None,
                 error, job_id, epoch),
            )
            self._conn.commit()
            return cur.rowcount > 0
