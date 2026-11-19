-- AlphaSAGE shares the existing owner-scoped durable mining task queue.
-- Replacing this constraint is safe to replay; no task or factor rows are changed.
BEGIN;

ALTER TABLE alpha_mining_task_specs
    DROP CONSTRAINT IF EXISTS alpha_mining_task_specs_algorithm_check;

ALTER TABLE alpha_mining_task_specs
    ADD CONSTRAINT alpha_mining_task_specs_algorithm_check
    CHECK (algorithm IN ('rdagent', 'statistical', 'alphasage'));

COMMIT;
