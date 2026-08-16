# Kernel-Evolving: Init, Backup & Fresh Commands Plan

## Problem Statement

Kernel-evolving has matured to a production-ready self-evolving agent (v1.8.3-evolving) with:
- Session memory persistence (chat_history_evolving.db, promoted_signals.db)
- Tool-access replicas (ADR-008 implemented)
- Inter-replica messaging pipeline
- Evolution dashboard on dedicated port (8779)

However, deployment and maintenance suffer from:
1. **Cross-contamination** between `kernel` and `kernel-evolving` workspaces (shared `.env`, overlapping ecosystem directories)
2. **No clean initialization** — first-run setup scattered across `install.sh`, `start.sh`, and manual steps
3. **No backup/restore** — migrating to new machines or recovering from corruption is manual
4. **No "fresh start"** — testing evolution from clean slate requires manual cleanup

## Proposed Commands

Add three new operational commands:

### 1. `/init` (or `/evolve init`)
**Purpose:** One-time initialization of kernel-evolving agent state.
**Actions:**
- Verify kernel base is running (port 8769 health)
- Create workspace directories (`~/.kernel-evolving/workspace/`, `~/.kernel/workspace/`)
- Initialize databases with schema if missing (`chat_history_evolving.db`, `promoted_signals.db`)
- Bootstrap ecosystem directories (community/private/third-party) with git clone of default repos
- Set up `.env` symlink to kernel base or create template
- Register with evolution dashboard
- Return structured status report

### 2. `/backup` (or `/evolve backup`)
**Purpose:** Create timestamped snapshot of kernel-evolving's entire state.
**Actions:**
- Compress into `.tar.gz` with timestamp: `kernel-evolving-backup-YYYY-MM-DD-HHMMSS.tar.gz`
- Include:
  - Both databases (`chat_history_evolving.db`, `promoted_signals.db`)
  - Workspace directory (`~/.kernel-evolving/workspace/`)
  - Ecosystem directories (`~/.kernel-evolving/ecosystem/`)
  - Config files (`config.yaml`, `.env`)
  - Current skill/routine definitions
- Store backup in `~/.kernel-evolving/backups/` (create directory)
- Log backup metadata (size, file count) to backup manifest
- Return backup path and SHA256 checksum

### 3. `/fresh` (or `/evolve fresh`)
**Purpose:** Reset to clean state while preserving previous state via backup.
**Actions:**
1. **Auto-backup:** Call `/backup` internally, store as pre-fresh snapshot
2. **Stop services:** Gracefully stop API (port 8779) if running
3. **Clean workspace:** Remove:
   - Database files (but keep backups)
   - Workspace contents (except `backups/` directory)
   - Temporary files (`/tmp/kernel_evolving_*`)
4. **Reinitialize:** Call `/init` to create fresh structures
5. **Restart API:** Start kernel-evolving on port 8779
6. **Report:** Provide backup location and fresh status

## Implementation Details

### Command Routing
**Option A:** Extend existing `/evolve` command family:
```
/evolve init
/evolve backup [--full]
/evolve fresh [--keep-ecosystem]
```

**Option B:** New slash commands:
```
/init
/backup
/fresh
```

**Recommended:** Option A for consistency with existing `/evolve` namespace.

### API Endpoints
Add to `src/api.py`:
- `POST /evolve/init` → returns `{"status": "initialized", "workspace": "...", "databases": ["..."]}`
- `POST /evolve/backup` → returns `{"backup_path": "...", "size_mb": X, "sha256": "..."}`
- `POST /evolve/fresh` → returns `{"backup_path": "...", "fresh": true, "api_restarted": true}`

### File Structure Additions
```
~/.kernel-evolving/
├── backups/
│   ├── 2026-05-10-153045/
│   │   ├── manifest.json
│   │   ├── workspace.tar.gz
│   │   └── databases.tar.gz
│   └── latest -> 2026-05-10-153045/
├── scripts/
│   ├── backup.sh
│   ├── fresh.sh
│   └── init.sh
└── workspace/  # recreated by /fresh
```

### Database Schema Verification
Add `src/db.py` with:
- `ensure_schema()` — CREATE TABLE IF NOT EXISTS for both databases
- `validate_schema()` — PRAGMA table_info compare against expected columns
- `vacuum_and_backup()` — SQLite VACUUM before backup

### Cross-Contamination Mitigation
**Problem:** Kernel and kernel-evolving share:
- `.env` file (symlink)
- `~/.kernel/workspace/` vs `~/.kernel-evolving/workspace/`
- Potential skill/routine name collisions

**Solutions:**
1. **Namespaced directories:** Ensure kernel-evolving uses `~/.kernel-evolving/` exclusively
2. **Env var prefixes:** `KERNEL_EVO_*` vs `KERNEL_*`
3. **Database distinct names:** `chat_history_evolving.db` already distinct
4. **Port separation:** 8769 (kernel) vs 8779 (evolving) already done

## Integration with Evolution Dashboard

The internal evolution dashboard (port 8779) should:
1. **Add "System" tab** showing:
   - Init status (last run, workspace size)
   - Backup history (list with restore buttons)
   - Fresh count (how many times reset)
2. **Automate backup schedule** (optional daily/weekly)
3. **Provide one-click "Fresh Start"** button that calls `/evolve/fresh`

## Backup Strategy

### What to backup (priority order):
1. **Critical:** Databases (chat history, promoted signals)
2. **Important:** Workspace files (user.json, thoughts/, ideas/)
3. **Optional:** Ecosystem skills/routines (git clones can be re-fetched)
4. **Config:** config.yaml, .env (redacted)

### What to exclude:
- Virtual environment (`.venv/` — can be rebuilt)
- Log files (`/tmp/kernel_evolving_api.log`)
- Cache directories
- Large model files (shared with kernel base)

### Retention policy:
- Keep last 7 daily backups
- Keep last 4 weekly backups
- Auto-cleanup via cron or `/evolve cleanup-backups`

## Fresh Command Workflow

```
User: /evolve fresh
↓
1. Check if evolution is active → pause if needed
2. Call internal backup (with "pre-fresh" label)
3. Stop API gracefully (send SIGTERM, wait 5s)
4. Remove:
   - ~/.kernel-evolving/workspace/* (except backups/)
   - ~/.kernel-evolving/workspace/chat_history_evolving.db
   - ~/.kernel-evolving/workspace/promoted_signals.db
   - /tmp/kernel_evolving_*
5. Recreate workspace directories
6. Initialize databases with schema
7. Start API (bind to 8779)
8. Seed with default thoughts if empty
9. Return: {
     "backup": "/path/to/backup.tar.gz",
     "fresh_init": true,
     "api_restarted": true,
     "dashboard_url": "http://localhost:8779/evolution"
   }
```

## Testing Considerations

### Test scenarios:
1. **/evolve init** on fresh install (no existing workspace)
2. **/evolve init** on existing workspace (should report already initialized)
3. **/evolve backup** during active evolution (should pause discovery thread)
4. **/evolve fresh** with active replicas (should warn/force stop)
5. **Restore from backup** (separate restore command to be implemented later)

### Integration tests:
- Backup → fresh → verify empty state
- Fresh → init → verify basic functionality
- Cross-contamination test: ensure kernel base unaffected

## Rollout Plan

### Phase 1: Core implementation (1-2 days)
- Add `src/db.py` with schema management
- Add `src/backup.py` with tar.gz creation
- Extend `src/api.py` with three new endpoints
- Update `src/agent.py` to handle `/evolve init|backup|fresh` commands
- Create `scripts/init.sh`, `scripts/backup.sh`, `scripts/fresh.sh`

### Phase 2: Dashboard integration (1 day)
- Add System tab to evolution dashboard
- Implement backup history display
- Add one-click buttons

### Phase 3: Testing & documentation (1 day)
- Write unit tests for backup/fresh cycles
- Update SKILL.md with new commands
- Create recovery guide

### Phase 4: Deployment
- Merge to `main` branch
- Tag as `v1.9.0-evolving`
- Update production instance
- Schedule regular backups

## Security Considerations

1. **.env redaction:** Backup should replace API keys/tokens with `[REDACTED]`
2. **File permissions:** Backup archives should be `chmod 600`
3. **No remote storage:** Backups stay local unless user explicitly exports
4. **Fresh confirmation:** Require explicit user approval (`/evolve fresh --confirm`)

## Future Extensions

1. **/evolve restore <backup-id>** — restore from specific backup
2. **Cloud backup** — optional upload to S3/Backblaze
3. **Migration tool** — kernel → kernel-evolving state transfer
4. **Health checks** — automated schema validation, repair

## Conclusion

Adding init, backup, and fresh commands addresses critical production readiness gaps:
- **Reproducible deployments** via `/evolve init`
- **Disaster recovery** via `/evolve backup`
- **Testing isolation** via `/evolve fresh`

These commands complement the existing evolution capabilities and provide the operational maturity needed for school demonstrations and production use.