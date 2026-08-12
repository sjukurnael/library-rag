-- Who is allowed to sign in.
--
-- The app gates on Google sign-in (web/auth.py). Google proves WHO someone is;
-- this table decides whether that person gets in. Both halves are needed --
-- anyone on earth has a Google account, so identity alone is not authorisation.
--
-- A table rather than an ALLOWED_EMAILS environment variable, because granting
-- access should not require a redeploy. Add a row (Supabase table editor, or
-- `python -m library_rag.cli.users --add`) and the next sign-in works.
--
-- Nothing here is Supabase-specific: it is a plain Postgres table read with
-- psycopg. Supabase happens to host the database and its dashboard happens to
-- be a convenient CRUD UI. Moving to any other Postgres changes nothing.

CREATE TABLE allowed_users (
    -- Lowercased on the way in AND on lookup. "Nael@Gmail.com" and
    -- "nael@gmail.com" are one Google account, and a capital letter should
    -- never be the reason someone cannot get in.
    email      TEXT PRIMARY KEY CHECK (email = lower(email)),
    -- Free text: "me", "dave - borrowed the Philemon commentary". Purely so a
    -- list of addresses six months from now is still readable.
    note       TEXT,
    added_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Deliberately NOT seeded. An empty table locks everyone out including the
-- owner, which is the correct default for a table that grants access -- a
-- seeded address would be a backdoor that survives into every deployment. The
-- bootstrap is `python -m library_rag.cli.users --add you@example.com`.
