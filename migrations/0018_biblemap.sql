-- The Bible map: places, people and events, and the links between them.
--
-- Its own schema, `biblemap`, unlike 0006 which put bible_verses in `public`.
-- That call was right for ONE table with an unmistakable name. This is six
-- tables, three of them named `places`, `people` and `events` -- names generic
-- enough that something else in this app could want them one day -- and none
-- of it shares state with the library. A schema says "this is a separate
-- product" in the one place anyone reads the database from.
--
-- Loaded, not written to: library_rag/biblemap.py TRUNCATEs and COPYs all of it
-- from the source spreadsheets in one transaction. The IDs are the source
-- data's own IDs (PlaceID, PersonID), so a row here can be traced straight back
-- to the file it came from.

CREATE SCHEMA biblemap;

CREATE TABLE biblemap.places (
    place_id  INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    alt_names TEXT NOT NULL DEFAULT '',
    -- NULL, not 0, when the source has no usable coordinate. The source
    -- spells "unknown" as 0/0 (Eden), blank, or a guess like "37?" (Luz), and
    -- a 0/0 that reached the map would put a pin in the Gulf of Guinea.
    lat       DOUBLE PRECISION,
    lng       DOUBLE PRECISION,
    comments  TEXT NOT NULL DEFAULT '',
    verses    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE biblemap.people (
    person_id    INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    alt_name     TEXT NOT NULL DEFAULT '',
    descriptor   TEXT NOT NULL DEFAULT '',
    subject_type TEXT NOT NULL DEFAULT '',
    gender       TEXT NOT NULL DEFAULT '',
    verses       TEXT NOT NULL DEFAULT ''
);

-- event_id is the event's row in the source spreadsheet, which is in Bible
-- order. So ORDER BY event_id IS the timeline, with no date column to invent:
-- only 74 of 3,688 people have a birth year, and a guessed chronology would be
-- a claim the data does not make.
CREATE TABLE biblemap.events (
    event_id    INTEGER PRIMARY KEY,
    passage     TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    icon        TEXT NOT NULL DEFAULT ''
);

-- Where it happened. A set: the source repeats an ID within a cell now and
-- then, and a place is either part of an event or not.
CREATE TABLE biblemap.event_places (
    event_id INTEGER NOT NULL REFERENCES biblemap.events ON DELETE CASCADE,
    place_id INTEGER NOT NULL REFERENCES biblemap.places,
    position SMALLINT NOT NULL,
    PRIMARY KEY (event_id, place_id)
);

-- The path travelled. A SEQUENCE, not a set: the key is the position, so a
-- route that returns to where it started (Samaria -> Damascus -> Samaria) keeps
-- both visits.
CREATE TABLE biblemap.event_route (
    event_id INTEGER NOT NULL REFERENCES biblemap.events ON DELETE CASCADE,
    position SMALLINT NOT NULL,
    place_id INTEGER NOT NULL REFERENCES biblemap.places,
    PRIMARY KEY (event_id, position)
);

CREATE TABLE biblemap.event_people (
    event_id  INTEGER NOT NULL REFERENCES biblemap.events ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES biblemap.people,
    position  SMALLINT NOT NULL,
    PRIMARY KEY (event_id, person_id)
);

-- The two reverse lookups the page is built on: "what happened here" and
-- "what did this person do". The primary keys already cover the forward
-- direction (an event's places, an event's people).
CREATE INDEX event_places_place_idx ON biblemap.event_places (place_id);
CREATE INDEX event_route_place_idx  ON biblemap.event_route (place_id);
CREATE INDEX event_people_person_idx ON biblemap.event_people (person_id);
