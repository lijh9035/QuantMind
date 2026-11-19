-- Owner-scoped AlphaSAGE rd_mined materialization generations.
-- This coexists with the legacy global qm_user_factor_* publishing tables.
BEGIN;

CREATE TABLE IF NOT EXISTS alpha_user_factor_generations (
    tenant_id TEXT NOT NULL CHECK (btrim(tenant_id) <> ''),
    user_id TEXT NOT NULL CHECK (btrim(user_id) <> ''),
    generation_id TEXT NOT NULL
        CHECK (length(generation_id) BETWEEN 1 AND 64),
    dataset_key TEXT NOT NULL DEFAULT 'rd_mined'
        CHECK (dataset_key = 'rd_mined'),
    status TEXT NOT NULL DEFAULT 'building'
        CHECK (status IN ('building', 'published', 'failed', 'archived')),
    job_id TEXT NOT NULL,
    manifest_json JSONB NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(manifest_json) = 'object'),
    row_count BIGINT NOT NULL DEFAULT 0 CHECK (row_count >= 0),
    byte_count BIGINT NOT NULL DEFAULT 0 CHECK (byte_count >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    published_at TIMESTAMPTZ,
    PRIMARY KEY (tenant_id, user_id, generation_id),
    UNIQUE (tenant_id, user_id, job_id),
    FOREIGN KEY (tenant_id, user_id, job_id)
        REFERENCES alpha_factor_jobs (tenant_id, user_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_user_factor_gen_owner_state
    ON alpha_user_factor_generations
        (tenant_id, user_id, dataset_key, status, created_at DESC);

CREATE TABLE IF NOT EXISTS alpha_user_factor_generation_features (
    tenant_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    feature_key TEXT NOT NULL
        CHECK (length(feature_key) BETWEEN 1 AND 128),
    factor_id TEXT NOT NULL CHECK (btrim(factor_id) <> ''),
    factor_code_sha256 TEXT NOT NULL
        CHECK (length(factor_code_sha256) = 64),
    factor_revision_no INTEGER NOT NULL CHECK (factor_revision_no >= 1),
    row_count BIGINT NOT NULL CHECK (row_count > 0),
    source_task_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, generation_id, feature_key),
    FOREIGN KEY (tenant_id, user_id, generation_id)
        REFERENCES alpha_user_factor_generations
            (tenant_id, user_id, generation_id)
);

CREATE INDEX IF NOT EXISTS idx_alpha_user_factor_features_owner_factor
    ON alpha_user_factor_generation_features (tenant_id, user_id, factor_id);

CREATE INDEX IF NOT EXISTS idx_alpha_user_factor_features_owner_source_task
    ON alpha_user_factor_generation_features (tenant_id, user_id, source_task_id)
    WHERE source_task_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS alpha_user_factor_dataset_active (
    tenant_id TEXT NOT NULL CHECK (btrim(tenant_id) <> ''),
    user_id TEXT NOT NULL CHECK (btrim(user_id) <> ''),
    dataset_key TEXT NOT NULL CHECK (dataset_key = 'rd_mined'),
    generation_id TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id, dataset_key),
    FOREIGN KEY (tenant_id, user_id, generation_id)
        REFERENCES alpha_user_factor_generations
            (tenant_id, user_id, generation_id)
);

COMMIT;
