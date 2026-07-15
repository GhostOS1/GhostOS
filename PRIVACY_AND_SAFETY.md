# GhostOS — Privacy and Safety

---

## 1. Data Handling

**What GhostOS collects/indexes:**
- File content and metadata from folders you connect (Desktop, Documents, Downloads, Pictures, Videos, Music, VS Code/git projects, or manually added folders)
- Local Chrome/Edge browser history, bookmarks, and downloads (read from local profile files, not live tabs)
- Foreground application activity (window titles, app names, timestamps) via `activity_tracker.py`
- Conversation history with the AI Assistant (in-memory, see §3)

**Where it's stored:** entirely in a local SQLite database (managed by `vectorstore.py`), on the same machine, alongside the embeddings generated for semantic search.

**What it's never used for:** there is no telemetry, analytics, or usage-reporting code anywhere in the backend — see [`LOCAL_AI_VERIFICATION.md`](LOCAL_AI_VERIFICATION.md) for the grep-verified network audit confirming this.

---

## 2. Permissions

GhostOS's action system (`action_agent.py` + `action_registry.py` + `action_permissions.py`) uses an **allowlist**, not open-ended OS access:

- The model can only invoke a fixed set of pre-approved actions: open file, open folder, open URL, open application (by trusted alias only, no arbitrary command line), create note, create folder.
- Every action is validated in `action_permissions.py` **before** touching the OS — nothing the model outputs is executed directly.
- **Prohibited executables are explicitly blocked**, even if somehow reachable through an allowlist misconfiguration: `cmd.exe`, `powershell.exe`, `pwsh.exe`, `wscript.exe`, `cscript.exe`, `mshta.exe`, `rundll32.exe`, `regsvr32.exe`, `wmic.exe`, and script-file suffixes (`.bat`, `.cmd`, `.ps1`, `.vbs`, `.js`, `.hta`, etc.).
- Note creation is confined to the user's home directory (or explicitly configured roots) — it cannot write files anywhere on the system.
- **There is no arbitrary shell execution path anywhere in the action system** — this is stated as a hard invariant in the code's own comments, not just a convention.

---

## 3. Storage

- All data lives in a local SQLite file — no cloud sync, no remote backup, no multi-device replication.
- **Conversation memory and session context** (last file/folder/topic discussed, used for pronoun resolution like "open it") is **in-process and non-persistent** — it lives in a Python-level global (`memory_agent.py`'s `_session_context`) and resets when the backend restarts. It is not written to disk.
- **Indexed content and embeddings** (from files, browser history, activity) *are* persisted to disk in SQLite, and will remain there — including after uninstalling the app, unless the database file is manually deleted — until GhostOS's own data-clearing endpoint (`/api/data/clear`) is used.
- **No encryption at rest is implemented** for the SQLite database. Anyone with local access to the machine (or the file, if copied off it) can read the indexed content, browser history, and activity log in plain form. This is a real limitation, not a minor one, given what's stored — see §4.
- Storage grows unbounded with usage; there is currently no automatic retention or pruning policy.

---

## 4. Limitations and Potential Risks

Stated plainly, not softened:

- **No encryption at rest.** Given GhostOS indexes browser history and file contents by design, the local database is a meaningful target if the machine itself is compromised, stolen, or shared. Full-disk encryption (BitLocker, etc.) mitigates this at the OS level, but GhostOS itself does not add its own layer.
- **The sensitive-path blacklist is pattern-based, not exhaustive.** `indexer.py`'s `SENSITIVE_PATTERNS` list (`login data`, `1password`, `keepass`, `wallet.dat`, `cookies`, `.ssh`, `private key`, `id_rsa`, `.env`, `credentials`) and `action_permissions.py`'s `SENSITIVE_COMPONENTS` (`.ssh`, `.gnupg`, `.aws`, `.azure`, `.kube`, password managers, `wallet.dat`, `keychain`, etc.) cover well-known cases but are string/component matches — a credential file with an unrecognized name would not be caught. This should be understood as **defense-in-depth, not a guarantee**.
- **Broad local data access by design.** Reading local browser history and tracking foreground-app activity are core to what GhostOS does — this is a genuine privacy tradeoff, not a bug: the tool's entire value proposition depends on seeing what the user has been doing. Anyone installing GhostOS should understand it as functionally equivalent to a local activity/history logger, scoped only by what folders and features they enable.
- **Single-user design, no access control.** Anyone with access to the machine/user account has full access to everything GhostOS has indexed — there's no separate authentication layer within the app itself.
- **Action allowlist reduces but does not eliminate risk.** The allowlist blocks shell execution and known-dangerous executables, but any allowlisted "open application" alias is only as safe as its own configuration — a misconfigured alias pointing at something unintended would bypass the *intent* of the safety design even though the *mechanism* (no arbitrary commands) still holds.
- **OCR/voice processing happens locally but on potentially sensitive content.** Scanned documents or audio processed through OCR/voice are never sent anywhere, but the extracted text is indexed the same as any other content — including anything sensitive in the original that wasn't caught by the blacklist.

---

## 5. Recommendations for Users (Practical, Not Just Disclosure)

- Use full-disk encryption (BitLocker on Windows) since GhostOS itself doesn't encrypt its database.
- Don't point GhostOS at folders containing highly sensitive material you wouldn't want indexed and semantically searchable, even locally — the blacklist is a safety net, not a substitute for folder-level judgment.
- Periodically use `/api/data/clear` (or equivalent settings-panel option) if you want to reset accumulated history rather than letting it grow indefinitely.
- Treat GhostOS's local database file with the same care as a password manager's vault file when it comes to backups, sharing, or transferring the machine to someone else.
