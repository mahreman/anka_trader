-- P2 Migration: Add ensemble fields to Decision table
--
-- This migration adds three new fields to the decisions table:
-- 1. p_up: Ensemble probability of upward price movement (0-1)
-- 2. disagreement: Analyst disagreement metric (0-1)
-- 3. analyst_signals: JSON string of individual analyst signals
--
-- Run this migration BEFORE using the P2 ensemble features.
--
-- Usage:
--   sqlite3 data/otonom_trader.db < scripts/migrate_p2_ensemble.sql

-- Add new columns to decisions table
ALTER TABLE decisions ADD COLUMN p_up REAL DEFAULT NULL;
ALTER TABLE decisions ADD COLUMN disagreement REAL DEFAULT NULL;
ALTER TABLE decisions ADD COLUMN analyst_signals TEXT DEFAULT NULL;

-- Verify the changes
SELECT
    name,
    type
FROM
    pragma_table_info('decisions')
WHERE
    name IN ('p_up', 'disagreement', 'analyst_signals');
