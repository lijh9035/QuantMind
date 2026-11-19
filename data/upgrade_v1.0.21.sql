-- Durable quota ledger used by owner-scoped local artifact writers.
-- New tables only; reconciliation opens admission after inventorying existing files.
BEGIN;

CREATE TABLE IF NOT EXISTS storage_quota_state (
    singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton = TRUE),
    ready BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
INSERT INTO storage_quota_state (singleton, ready)
VALUES (TRUE, FALSE)
ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS storage_quota_accounts (
    tenant_id TEXT NOT NULL CHECK (btrim(tenant_id) <> ''),
    user_id TEXT NOT NULL CHECK (btrim(user_id) <> ''),
    used_bytes BIGINT NOT NULL DEFAULT 0 CHECK (used_bytes >= 0),
    reserved_bytes BIGINT NOT NULL DEFAULT 0 CHECK (reserved_bytes >= 0),
    quota_bytes BIGINT NOT NULL CHECK (quota_bytes >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE IF NOT EXISTS storage_quota_objects (
    resource_key TEXT PRIMARY KEY CHECK (btrim(resource_key) <> ''),
    tenant_id TEXT NOT NULL CHECK (btrim(tenant_id) <> ''),
    user_id TEXT NOT NULL CHECK (btrim(user_id) <> ''),
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS storage_quota_objects_owner
    ON storage_quota_objects (tenant_id, user_id);

CREATE TABLE IF NOT EXISTS storage_quota_reservations (
    reservation_id TEXT PRIMARY KEY CHECK (btrim(reservation_id) <> ''),
    tenant_id TEXT NOT NULL CHECK (btrim(tenant_id) <> ''),
    user_id TEXT NOT NULL CHECK (btrim(user_id) <> ''),
    resource_key TEXT NOT NULL CHECK (btrim(resource_key) <> ''),
    reserved_bytes BIGINT NOT NULL CHECK (reserved_bytes >= 0),
    replaced_bytes BIGINT NOT NULL DEFAULT 0 CHECK (replaced_bytes >= 0),
    actual_bytes BIGINT CHECK (actual_bytes IS NULL OR actual_bytes >= 0),
    state TEXT NOT NULL CHECK (state IN ('reserved', 'settled', 'released')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS storage_quota_reservations_owner_state
    ON storage_quota_reservations (tenant_id, user_id, state, updated_at);
CREATE INDEX IF NOT EXISTS storage_quota_reservations_resource_state
    ON storage_quota_reservations (resource_key, state, updated_at);

COMMIT;
