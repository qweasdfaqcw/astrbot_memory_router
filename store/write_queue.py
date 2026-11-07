"""SQLite 异步写入队列。

所有写操作进入单消费者队列。消费者一次取若干 job，在同一个事务里执行，
每个 job 使用 SAVEPOINT 隔离错误，最后统一 commit：
- 避免 SQLite 写锁竞争；
- 批量提交降低 fsync 次数；
- 单个 job 失败不会拖垮同批其他 job；
- ``await submit()`` 返回时数据已提交，满足 write_epoch 的「写完即可见」。
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class _Job(Generic[T]):
    func: Callable[[sqlite3.Connection], T]
    future: asyncio.Future[T]


class WriteQueue:
    """单消费者、批量提交的 SQLite 写队列。"""

    def __init__(self, db_path: Path, *, batch_size: int = 32, batch_wait: float = 0.01):
        self.db_path = Path(db_path)
        self.batch_size = max(1, batch_size)
        self.batch_wait = max(0.0, batch_wait)
        self._queue: asyncio.Queue[_Job[Any]] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._connection: sqlite3.Connection | None = None
        self._start_lock = asyncio.Lock()
        self._pending = 0
        self._closed = False

    async def _start(self) -> None:
        if self._closed:
            raise RuntimeError("write queue is closed")
        if self._worker and not self._worker.done():
            return
        async with self._start_lock:
            if self._worker and not self._worker.done():
                return
            self._worker = asyncio.create_task(self._run(), name="memory-router-write-queue")

    async def submit(self, func: Callable[[sqlite3.Connection], T]) -> T:
        """提交写操作并等待提交完成。"""
        await self._start()
        loop = asyncio.get_running_loop()
        job: _Job[T] = _Job(func=func, future=loop.create_future())
        self._pending += 1
        await self._queue.put(job)
        try:
            return await job.future
        finally:
            self._pending = max(0, self._pending - 1)

    async def _run(self) -> None:
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            self._connection = conn
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=10000")
        except Exception as exc:  # noqa: BLE001
            self._connection = None
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            # 把启动失败回传给所有排队/后续调用，避免 future 永久挂起。
            while True:
                try:
                    job = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not job.future.done():
                    job.future.set_exception(exc)
                self._queue.task_done()
            self._worker = None
            return
        try:
            while True:
                first = await self._queue.get()
                batch = [first]
                if self.batch_size > 1 and self.batch_wait > 0:
                    try:
                        more = await asyncio.wait_for(
                            self._queue.get(), timeout=self.batch_wait
                        )
                        batch.append(more)
                    except asyncio.TimeoutError:
                        pass
                while len(batch) < self.batch_size:
                    try:
                        batch.append(self._queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                self._execute_batch(conn, batch)
                for _ in batch:
                    self._queue.task_done()
        except asyncio.CancelledError:
            # 关闭时把尚未执行的 job 明确失败，避免调用方永久等待。
            while True:
                try:
                    job = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not job.future.done():
                    job.future.set_exception(RuntimeError("write queue closed"))
                self._queue.task_done()
            raise
        finally:
            self._connection = None
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

    def _execute_batch(self, conn: sqlite3.Connection, batch: list[_Job[Any]]) -> None:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
            return

        succeeded: list[tuple[_Job[Any], Any]] = []
        try:
            for index, job in enumerate(batch):
                savepoint = f"job_{index}"
                try:
                    conn.execute(f"SAVEPOINT {savepoint}")
                    result = job.func(conn)
                    conn.execute(f"RELEASE {savepoint}")
                    succeeded.append((job, result))
                except Exception as exc:  # noqa: BLE001 - 写入失败必须回传给调用方
                    try:
                        conn.execute(f"ROLLBACK TO {savepoint}")
                        conn.execute(f"RELEASE {savepoint}")
                    except sqlite3.Error:
                        pass
                    if not job.future.done():
                        job.future.set_exception(exc)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            try:
                conn.rollback()
            except sqlite3.Error:
                pass
            for job, _ in succeeded:
                if not job.future.done():
                    job.future.set_exception(exc)
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
        else:
            # 只有 commit 成功后才把结果交给调用方，避免“返回成功但已回滚”。
            for job, result in succeeded:
                if not job.future.done():
                    job.future.set_result(result)

    async def flush(self) -> None:
        """等待所有已提交 job 完成。"""
        while self._pending > 0 or not self._queue.empty():
            await asyncio.sleep(0.005)

    async def close(self) -> None:
        if self._closed:
            return
        await self.flush()
        self._closed = True
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._worker = None
