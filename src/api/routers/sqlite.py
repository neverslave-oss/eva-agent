"""api/routers/sqlite.py — sqlite inspection/editing routes.

Extracted from src/api/__init__.py (issue #2, plan 2026-10-03). Route functions
resolve live api state through the _api() seam; _safe_workspace_path from
api.helpers. Behavior-identical; re-exported from api.__init__.
"""
import sys
import importlib
import sqlite3

from fastapi import APIRouter, Body as _Body
from fastapi.responses import JSONResponse

from ..helpers import _safe_workspace_path

router = APIRouter()


def _api():
    m = sys.modules.get("api")
    if m is None:  # pragma: no cover - defensive
        m = importlib.import_module("api")
    return m


@router.get("/sqlite/tables")
def sqlite_tables(path: str = ""):
    """List all tables in a SQLite database file."""
    _m = _api()
    import pathlib, sqlite3
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists() or not fp.is_file():
        return JSONResponse({"error": "file not found"}, status_code=404)
    ext = fp.suffix.lower()
    if ext not in (".sqlite", ".sqlite3", ".db", ".db3"):
        return JSONResponse({"error": "not a SQLite file"}, status_code=400)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [r["name"] for r in tables]
        conn.close()
        return {"tables": table_names, "count": len(table_names), "path": path}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.get("/sqlite/table")
def sqlite_table(path: str = "", table: str = "", page: int = 1, per_page: int = 100):
    """Get table schema and paginated data from a SQLite database."""
    _m = _api()
    import pathlib, sqlite3
    if not path or not table:
        return JSONResponse({"error": "path and table are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists() or not fp.is_file():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.row_factory = sqlite3.Row
        # Verify table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,)
        ).fetchone()
        if not exists:
            conn.close()
            return JSONResponse({"error": f"table '{table}' not found"}, status_code=404)
        # Schema
        schema = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        columns = [{"cid": r["cid"], "name": r["name"], "type": r["type"],
                     "notnull": bool(r["notnull"]), "pk": bool(r["pk"])} for r in schema]
        # Row count
        count_row = conn.execute(f"SELECT COUNT(*) as cnt FROM '{table}'").fetchone()
        total_rows = count_row["cnt"] if count_row else 0
        # Paginated data
        offset = (max(1, page) - 1) * per_page
        rows_raw = conn.execute(
            f"SELECT * FROM '{table}' LIMIT ? OFFSET ?", (per_page, offset)
        ).fetchall()
        col_names = [c["name"] for c in schema]
        rows_data = []
        for r in rows_raw:
            row_dict = {}
            for cn in col_names:
                val = r[cn]
                # Convert bytes to hex string for display
                if isinstance(val, bytes):
                    val = "<bytes>"
                row_dict[cn] = val
            rows_data.append(row_dict)
        conn.close()
        return {
            "columns": columns,
            "column_names": col_names,
            "rows": rows_data,
            "total_rows": total_rows,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total_rows + per_page - 1) // per_page),
            "path": path,
            "table": table,
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.put("/sqlite/row")
def sqlite_row_update(body: dict = _Body(...)):
    """Update a single row in a SQLite table. Body: {path, table, pk_column, pk_value, updates: {col: val}}"""
    _m = _api()
    import pathlib, sqlite3
    path = (body.get("path") or "").strip()
    table = (body.get("table") or "").strip()
    pk_column = (body.get("pk_column") or "").strip()
    pk_value = body.get("pk_value")
    updates = body.get("updates") or {}
    if not path or not table or not pk_column or pk_value is None or not updates:
        return JSONResponse({"error": "path, table, pk_column, pk_value, and updates are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.row_factory = sqlite3.Row
        set_clauses = []
        params = []
        for col, val in updates.items():
            set_clauses.append(f'"{col}" = ?')
            params.append(val)
        params.append(pk_value)
        sql = f'UPDATE "{table}" SET {", ".join(set_clauses)} WHERE "{pk_column}" = ?'
        conn.execute(sql, params)
        conn.commit()
        conn.close()
        return {"ok": True, "rows_affected": conn.total_changes}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.delete("/sqlite/row")
def sqlite_row_delete(path: str = "", table: str = "", pk_column: str = "", pk_value: str = ""):
    """Delete a single row from a SQLite table."""
    _m = _api()
    import pathlib, sqlite3
    if not path or not table or not pk_column or not pk_value:
        return JSONResponse({"error": "path, table, pk_column, and pk_value are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.execute(f'DELETE FROM "{table}" WHERE "{pk_column}" = ?', (pk_value,))
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return {"ok": True, "rows_deleted": affected}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.delete("/sqlite/table/data")
def sqlite_table_clear(path: str = "", table: str = ""):
    """Delete all data from a table while keeping its structure (DELETE FROM)."""
    _m = _api()
    import pathlib, sqlite3
    if not path or not table:
        return JSONResponse({"error": "path and table are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        conn.execute(f'DELETE FROM "{table}"')
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return {"ok": True, "rows_deleted": affected, "table": table}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.delete("/sqlite/db/data")
def sqlite_db_clear(path: str = ""):
    """Delete all data from all tables in a SQLite database while keeping structure."""
    _m = _api()
    import pathlib, sqlite3
    if not path:
        return JSONResponse({"error": "path is required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        total = 0
        cleared = []
        for t in tables:
            tname = t[0]
            affected = conn.execute(f'DELETE FROM "{tname}"').rowcount
            total += affected
            cleared.append(tname)
        conn.commit()
        conn.close()
        return {"ok": True, "tables_cleared": cleared, "total_rows_deleted": total}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

@router.post("/sqlite/row")
def sqlite_row_insert(body: dict = _Body(...)):
    """Insert a new row into a SQLite table. Body: {path, table, row: {col: val}}"""
    _m = _api()
    import pathlib, sqlite3
    path = (body.get("path") or "").strip()
    table = (body.get("table") or "").strip()
    row_data = body.get("row") or {}
    if not path or not table or not row_data:
        return JSONResponse({"error": "path, table, and row are required"}, status_code=400)
    full = _safe_workspace_path(path)
    if full is None:
        return JSONResponse({"error": "invalid or unsafe path"}, status_code=400)
    fp = pathlib.Path(full)
    if not fp.exists():
        return JSONResponse({"error": "file not found"}, status_code=404)
    try:
        conn = sqlite3.connect(str(fp), timeout=5)
        cols = ", ".join(f'"{c}"' for c in row_data.keys())
        placeholders = ", ".join("?" for _ in row_data)
        vals = list(row_data.values())
        sql = f'INSERT INTO "{table}" ({cols}) VALUES ({placeholders})'
        conn.execute(sql, vals)
        conn.commit()
        conn.close()
        return {"ok": True, "inserted": True}
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
