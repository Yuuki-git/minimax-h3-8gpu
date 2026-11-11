"""Workspace-local Triton cache for a filesystem that rejects rename/replace.

Selected through TRITON_CACHE_MANAGER. Vendor Triton remains read-only.
Content-addressed blobs are immutable after publication. A locked JSON index
commits each logical entry only after the blob is closed and verified; readers
reject an interrupted index or incomplete blob. No rename operation is used.
"""

from contextlib import contextmanager
import errno
import hashlib
import json
import os
from pathlib import Path
import threading

from triton.runtime.cache import CacheManager


_THREAD_LOCKS = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _component(value):
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or any(char in value for char in ("/", "\\", ":", "\0"))):
        raise ValueError("Cache keys and filenames must be single path components")
    return value


def _digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class WorkspaceCacheManager(CacheManager):
    def __init__(self, key, override=False, dump=False):
        self.key = _component(key)
        variable = "TRITON_DUMP_DIR" if dump else "TRITON_OVERRIDE_DIR" if override else "TRITON_CACHE_DIR"
        scope = Path(os.environ["SS5_ROUND1_CACHE_SCOPE"]).resolve(strict=True)
        root = Path(os.environ[variable]).resolve()
        if not root.is_relative_to(scope):
            raise ValueError("Triton cache root escaped the current run")
        directory = (root / self.key).resolve()
        if not directory.is_relative_to(scope) or directory == scope:
            raise ValueError("Triton cache key escaped the current run")
        directory.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.cache_dir = str(directory)
        self._directory = directory
        self.lock_path = str(directory / ".ss5-cache.lock")
        with _THREAD_LOCKS_GUARD:
            self._thread_lock = _THREAD_LOCKS.setdefault(self.cache_dir, threading.RLock())

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            if Path(self.lock_path).is_symlink():
                raise ValueError("Cache lock must not be a symlink")
            with os.fdopen(os.open(self.lock_path, flags, 0o660), "r+b") as lock:
                if os.name == "nt":
                    # Also permits genuine local concurrency regression on Windows.
                    import msvcrt
                    if os.fstat(lock.fileno()).st_size == 0:
                        lock.write(b"\0")
                        lock.flush()
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                    try:
                        yield
                    finally:
                        lock.seek(0)
                        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)

    def _entry_path(self, filename):
        _component(filename)
        key = hashlib.sha256(filename.encode("utf-8")).hexdigest()
        return self._directory / f".ss5-entry-{key}.json"

    def _blob_name(self, filename, digest):
        # Preserve the extension so Python's native extension loader recognizes .so.
        return f"ss5-{digest}{Path(filename).suffix}"

    def _write_complete(self, path, data):
        if path.is_symlink() or path.parent != self._directory:
            raise ValueError("Cache write escaped its directory")
        flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags, 0o660), "wb") as output:
            output.write(data)
            output.flush()
            try:
                os.fsync(output.fileno())
            except OSError as exc:
                if exc.errno not in {errno.EINVAL, errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}:
                    raise

    def _get_file_locked(self, filename):
        entry = self._entry_path(filename)
        if entry.is_symlink() or not entry.is_file():
            return None
        try:
            record = json.loads(entry.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or record.get("version") != 1 or record.get("filename") != filename:
                return None
            digest, size = record.get("sha256"), record.get("bytes")
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)
                    or not isinstance(size, int) or size < 0):
                return None
            expected = self._blob_name(filename, digest)
            if record.get("blob") != expected:
                return None
            path = self._directory / expected
            if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
                return None
            if _digest_file(path) != digest:
                return None
            return str(path)
        except (OSError, ValueError, TypeError):
            return None

    def get_file(self, filename):
        _component(filename)
        with self._locked():
            return self._get_file_locked(filename)

    def has_file(self, filename):
        return self.get_file(filename) is not None

    def _put_locked(self, data, filename):
        _component(filename)
        data = bytes(data) if isinstance(data, (bytes, bytearray, memoryview)) else str(data).encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        path = self._directory / self._blob_name(filename, digest)
        if path.is_symlink():
            raise ValueError("Cache artifact must not be a symlink")
        # Never truncate a published blob: other loaders may still map its bytes.
        if not path.is_file() or path.stat().st_size != len(data) or _digest_file(path) != digest:
            self._write_complete(path, data)
        if path.stat().st_size != len(data) or _digest_file(path) != digest:
            raise OSError("Cache artifact verification failed before publication")
        record = {"version": 1, "filename": filename, "blob": path.name,
                  "bytes": len(data), "sha256": digest}
        self._write_complete(self._entry_path(filename), (json.dumps(record) + "\n").encode("utf-8"))
        return str(path)

    def put(self, data, filename, binary=True):
        with self._locked():
            return self._put_locked(data, filename)

    def put_group(self, filename, group):
        _component(filename)
        with self._locked():
            for name, path in group.items():
                if self._get_file_locked(name) != str(path):
                    raise ValueError("Cache group contains an uncommitted or external artifact")
            value = json.dumps({"child_paths": {key: str(value) for key, value in group.items()}})
            return self._put_locked(value, f"__grp__{filename}")

    def get_group(self, filename):
        _component(filename)
        with self._locked():
            path = self._get_file_locked(f"__grp__{filename}")
            if path is None:
                return None
            try:
                record = json.loads(Path(path).read_text(encoding="utf-8"))
                children = record.get("child_paths")
                if not isinstance(children, dict):
                    return None
                for name, expected in children.items():
                    if self._get_file_locked(name) != expected:
                        return None
                return children
            except (OSError, ValueError, TypeError):
                return None
