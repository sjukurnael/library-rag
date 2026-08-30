-- Who ran it.
--
-- Runs are shared history now that Activity exists: anyone signed in sees every
-- run, and "who asked this" is the first thing a second reader wants to know.
-- classrooms has carried owner_email since 0012, so this is the same fact
-- stored the same way.
--
-- NULLABLE, for two reasons that both matter. Every run recorded before this
-- migration has no owner and never will -- backfilling a guess would be worse
-- than an honest blank. And config.auth_enabled() is false whenever
-- GOOGLE_CLIENT_ID is unset, which is how the app is run locally for UI work;
-- current_user() returns None there, so a NOT NULL column would make the
-- librarian unusable on a developer's machine. Same call 0013 made when it
-- relaxed classrooms.owner_email for exactly this case.
--
-- No index. Activity lists newest-first across everyone, and filtering by owner
-- is not a feature anyone has asked for; (agent, started_at DESC) from 0016 is
-- still the access path.

ALTER TABLE research_runs ADD COLUMN owner_email TEXT;
