-- Run only if upgrading an existing PostgreSQL volume from the old lab.
ALTER TABLE orders ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'pending';
UPDATE orders SET status = CASE WHEN predicted_price IS NULL THEN 'failed' ELSE 'priced' END WHERE status IS NULL;
