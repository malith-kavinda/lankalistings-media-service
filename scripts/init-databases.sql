-- Runs once, on first initialisation of the postgres volume.
-- The test suite truncates every table between tests, so it must never share a database with
-- development data.
CREATE DATABASE lankalistings_test OWNER lankalistings;
