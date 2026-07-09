"""Job manager para tareas largas en background.

Soporta dos modos:
- Thread-based: para jobs ligeros (I/O bound). El target corre en un hilo daemon.
- Process-based: para jobs CPU-intensivos (genetic optimizer, backtests masivos).
  El target corre en un PROCESO SEPARADO con su propio GIL, para no bloquear
  el event loop de FastAPI/uvicorn.

Comunicación proceso → padre vía multiprocessing.Queue (leída por un hilo
ligero que solo hace queue.get(), operación I/O que libera el GIL).
"""
from __future__ import annotations

import multiprocessing as mp
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Job:
    job_id: str
    job_type: str
    status: str = "pending"  # pending → running → completed | failed | cancelled
    progress: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    _thread: threading.Thread | None = field(default=None, repr=False)
    _process: mp.Process | None = field(default=None, repr=False)
    _cancel_flag: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        start = self.started_at or self.created_at
        return round(end - start, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
            "created_at": self.created_at,
        }


class JobManager:
    """Gestor de jobs en background thread-safe.

    Usa process-based execution para jobs CPU-intensivos (evita GIL contention
    con el event loop de FastAPI).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(
        self,
        job_type: str,
        target: Callable,
        *args,
        **kwargs,
    ) -> str:
        """Lanza un job en background (thread-based). Retorna el job_id.

        Para jobs CPU-intensivos, usa submit_process() en su lugar.
        """
        job_id = str(uuid.uuid4())[:12]
        job = Job(job_id=job_id, job_type=job_type, status="pending")

        def _runner():
            job.status = "running"
            job.started_at = time.time()
            try:
                result = target(job=job, *args, **kwargs)
                job.result = result
                job.status = "completed"
            except Exception as e:
                job.error = str(e)
                job.status = "failed"
            finally:
                job.finished_at = time.time()

        thread = threading.Thread(target=_runner, daemon=True, name=f"job-{job_id}")
        job._thread = thread

        with self._lock:
            self._jobs[job_id] = job
        thread.start()
        return job_id

    def submit_process(
        self,
        job_type: str,
        target: Callable,
        *args,
        **kwargs,
    ) -> str:
        """Lanza un job en un PROCESO SEPARADO (para jobs CPU-intensivos).

        Si multiprocessing falla (restricciones de la nube, RAM limitada),
        cae automáticamente a thread-based como fallback.
        """
        job_id = str(uuid.uuid4())[:12]
        job = Job(job_id=job_id, job_type=job_type, status="pending")

        try:
            # Cola para comunicación proceso → padre
            ctx = mp.get_context("spawn")  # Windows-safe
            queue: mp.Queue = ctx.Queue()

            # Lanzar el proceso hijo
            proc = ctx.Process(
                target=target,
                args=(queue,) + args,
                kwargs=kwargs,
                daemon=True,
                name=f"proc-{job_id}",
            )
            job._process = proc

            with self._lock:
                self._jobs[job_id] = job

            proc.start()
            self._start_queue_reader(job, queue, proc)
        except Exception:
            # Fallback: si multiprocessing falla (cloud, RAM), usar thread
            return self._submit_process_fallback(job, target, args, kwargs)

        return job_id

    def _start_queue_reader(self, job: Job, queue: mp.Queue, proc: mp.Process):
        """Hilo ligero que lee la cola y actualiza el job (NO hace CPU work)."""
        def _queue_reader():
            job.started_at = time.time()
            job.status = "running"
            try:
                while True:
                    if not proc.is_alive():
                        # Proceso murió sin dejar mensaje → abortar
                        raise TimeoutError("Worker process died unexpectedly")
                    msg = queue.get(timeout=15)  # I/O wait, libera GIL
                    msg_type = msg.get("type", "")

                    if msg_type == "started":
                        job.status = "running"
                    elif msg_type == "progress":
                        job.progress = msg
                    elif msg_type == "completed":
                        job.result = msg.get("result")
                        job.status = "completed"
                        break
                    elif msg_type == "failed":
                        job.error = msg.get("error", "Error desconocido")
                        job.status = "failed"
                        break
            except Exception:
                # Timeout o proceso muerto
                if proc.is_alive():
                    proc.terminate()
                if job.status not in ("completed", "failed", "cancelled"):
                    job.error = "Job timeout o proceso terminado inesperadamente"
                    job.status = "failed"
            finally:
                job.finished_at = time.time()
                if proc.is_alive():
                    proc.join(timeout=5)
                queue.close()
                queue.join_thread()

        reader_thread = threading.Thread(target=_queue_reader, daemon=True, name=f"reader-{job.job_id}")
        job._thread = reader_thread
        reader_thread.start()

    def _submit_process_fallback(self, job: Job, target: Callable, args, kwargs):
        """Fallback a thread-based si multiprocessing no está disponible.

        Ejecuta el target directamente en un thread, pasando un dummy queue
        que captura los mensajes de progreso.
        """
        import queue as queue_module

        dummy_queue: queue_module.Queue = queue_module.Queue()
        job_id = job.job_id

        def _drain_queue():
            """Lee todos los mensajes pendientes de la dummy queue."""
            while True:
                try:
                    msg = dummy_queue.get_nowait()
                    if msg.get("type") == "completed":
                        job.result = msg.get("result")
                        job.status = "completed"
                    elif msg.get("type") == "failed":
                        job.error = msg.get("error")
                        job.status = "failed"
                    elif msg.get("type") == "progress":
                        job.progress = msg
                except queue_module.Empty:
                    break

        def _thread_runner():
            job.started_at = time.time()
            job.status = "running"
            try:
                target(dummy_queue, *args, **kwargs)
            except Exception as e:
                _drain_queue()
                if job.status not in ("completed", "failed"):
                    job.error = str(e)
                    job.status = "failed"
            else:
                _drain_queue()
                if job.status not in ("completed", "failed", "cancelled"):
                    job.status = "completed"
            finally:
                job.finished_at = time.time()

        thread = threading.Thread(target=_thread_runner, daemon=True, name=f"job-fb-{job_id}")
        job._thread = thread

        with self._lock:
            self._jobs[job_id] = job
        thread.start()
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_status(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job is None:
            return None
        return job.to_dict()

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status not in ("pending", "running"):
            return False
        job._cancel_flag.set()
        job.status = "cancelled"
        job.finished_at = time.time()
        # Terminar el proceso hijo si existe
        if job._process and job._process.is_alive():
            job._process.terminate()
        return True

    def _cleanup_old_jobs(self, max_age_hours: float = 2.0) -> None:
        """Elimina jobs completados hace más de max_age_hours."""
        now = time.time()
        cutoff = now - max_age_hours * 3600
        with self._lock:
            to_remove = [
                jid for jid, j in self._jobs.items()
                if j.status in ("completed", "failed", "cancelled")
                and j.finished_at and j.finished_at < cutoff
            ]
            for jid in to_remove:
                del self._jobs[jid]


# Instancia global compartida por todos los routers
job_manager = JobManager()
