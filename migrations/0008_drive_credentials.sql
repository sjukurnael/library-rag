-- Where the Drive token lives when there is no filesystem to keep it on.
--
-- Cloud Run's disk is in-memory and discarded on every cold start, so a
-- token.json written there survives minutes on a scale-to-zero service.
-- Postgres is the only durable thing in this system.
--
-- ONE ROW, enforced by the CHECK below. One shared credential, not one per
-- user: everyone who can sign in to this app is looking at the SAME shared
-- Drive folder, so a second person's token would reach nothing new while
-- adding their entire Drive to what a compromise of this database exposes.
-- (`drive.readonly` is not folder-scoped -- Google offers no such scope -- so
-- whoever connects grants read access to all of their own Drive.)
--
-- Whoever connects, everyone benefits: the token is refreshed by whichever
-- allowlisted person happens to click Reconnect, which matters because the
-- OAuth consent screen behind it is in Testing status and Google revokes those
-- refresh tokens every 7 days.

CREATE TABLE drive_credentials (
    -- Singleton. The CHECK makes a second row impossible rather than merely
    -- discouraged, so no code has to decide which of two rows is "current".
    id           SMALLINT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    -- The full authorized-user JSON google-auth writes: access token, refresh
    -- token, client id/secret, scopes, expiry. Stored whole rather than split
    -- into columns because it is google-auth's format to define, not ours.
    token_json   TEXT NOT NULL,
    -- Which allowlisted email completed the consent. Not for authorisation --
    -- it is so the UI can say "connected by stephen@… 6 days ago" and whoever
    -- reads that knows whose Drive access is being used.
    connected_by TEXT NOT NULL,
    connected_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
