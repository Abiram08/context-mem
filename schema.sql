-- ContextVault schema (SQLite, local-first)
-- Tuned for coding-agent memory on small codebases (< ~50k LOC).
-- Swap path: implement MemoryStore against Postgres/Supabase later via storage.py.

CREATE TABLE IF NOT EXISTS agents (
  agent_id TEXT PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  agent_id TEXT NOT NULL REFERENCES agents(agent_id),
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_active_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  agent_id TEXT NOT NULL REFERENCES agents(agent_id),
  session_id TEXT REFERENCES sessions(session_id),
  content TEXT NOT NULL,
  -- Coding-memory kind: decision | convention | bug | file_map | api |
  -- preference | gotcha | architecture | todo | context
  kind TEXT NOT NULL DEFAULT 'context',
  tags TEXT,                 -- JSON array, e.g. ["auth","jwt"]
  paths TEXT,                -- JSON array of related file paths
  symbols TEXT,              -- JSON array of functions/classes/modules
  importance INTEGER NOT NULL DEFAULT 3 CHECK (importance BETWEEN 1 AND 5),
  -- Denormalized FTS document: content + kind + tags + paths + symbols
  search_doc TEXT NOT NULL DEFAULT '',
  access_count INTEGER NOT NULL DEFAULT 0,
  last_accessed_at TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(agent_id, kind);
CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(agent_id, access_count DESC);

-- Immutable audit trail of every memory change (mem0/Zep-style history):
-- lets agents (and developers) see what changed since the last session.
CREATE TABLE IF NOT EXISTS memory_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id INTEGER NOT NULL,
  agent_id TEXT NOT NULL,
  session_id TEXT,
  action TEXT NOT NULL CHECK (action IN ('created', 'updated', 'deleted', 'merged')),
  content_before TEXT,
  content_after TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memory_events_agent ON memory_events(agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memory_events_memory ON memory_events(memory_id);

-- Full-text index over the denormalized search document
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  search_doc,
  content='memories',
  content_rowid='id',
  tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, search_doc) VALUES (new.id, new.search_doc);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, search_doc)
    VALUES ('delete', old.id, old.search_doc);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, search_doc)
    VALUES ('delete', old.id, old.search_doc);
  INSERT INTO memories_fts(rowid, search_doc) VALUES (new.id, new.search_doc);
END;
