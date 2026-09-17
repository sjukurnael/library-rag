"""The Bible map: biblical places, people and events, and where each event happened.

The data is a hand-curated set of spreadsheets kept outside this repo (places
and people exported from Notion, events from Logos with IDs resolved against
both). This module turns those files into the `biblemap` schema and reads it
back for the page.

Three stages, the same split as bible.py and for the same reason -- each is
understandable and testable on its own:

    read_sources()   files -> Dataset          (no database)
    validate()       Dataset -> list of problems
    load()           Dataset -> Postgres       (no filesystem)

Alongside the parsed tables, every source file is also kept VERBATIM -- every
row, every column, as text -- in biblemap.source_files / source_rows. That copy
is what the "Show in original data" viewer reads: when the map looks wrong, the
first question is whether the spreadsheet says so, and answering it should not
mean opening Excel and hunting for a row.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

# The file names inside the source directory. Globs for the two Notion exports,
# because Notion appends a hash to every export ("Places Data 17ed4e..._all.csv")
# and re-exporting changes it.
PLACES_GLOB = "Places-Data/Places Data *_all.csv"
NEW_PLACES_FILE = "Places-Data/New Places (added during resolution).csv"
PEOPLE_GLOB = "People Data/People Data *_all.csv"
EVENTS_FILE = "Bible Events with Resolved PersonIDs and PlaceIDs.xlsx"

# The columns the map actually reads, per source file. The viewer marks these,
# so it is obvious which of the 43 People columns can change what the map shows.
USED_COLUMNS = {
    "events": ["Passage (Logos Data)", "Title", "Description", "Icon",
               "PlaceID", "RoutePlaceID", "resolved_people_ids"],
    "places": ["PlaceID", "PlaceName", "AltName", "AltSpelling", "Lat", "Lng",
               "Comments", "Verses"],
    "new_places": ["PlaceID", "PlaceName", "Lat", "Lng", "Comments"],
    "people": ["PersonID", "Name", "AltName", "Descriptor", "SubjectType",
               "Gender", "Verses"],
}

# Columns whose cells are lists of IDs pointing into another file -- the
# foreign keys between the spreadsheets. The viewer turns each ID into a link to
# the row it names. "Duplicate IDs" is not read by the map, but it is the list of
# candidate people the resolution chose between, which is exactly what someone
# checking a resolution wants to click through.
LINK_COLUMNS = {
    "events": {"PlaceID": "place", "RoutePlaceID": "place",
               "resolved_people_ids": "person", "Duplicate IDs": "person"},
}

# Which source files a map record can come from, for locate().
RECORD_SOURCES = {
    "event": ["events"],
    "place": ["places", "new_places"],
    "person": ["people"],
}


@dataclass
class Source:
    """One source file, verbatim.

    `rows` are (row_num, record_id, cells). row_num is the row as a spreadsheet
    program numbers it -- the header is row 1 -- so "row 41" in the viewer is
    row 41 when the file is opened in Excel. record_id is the event, place or
    person the row became, or None for a row that became nothing (a blank row,
    a place with no ID).
    """
    key: str
    filename: str
    columns: list
    rows: list = field(default_factory=list)

    def records(self):
        for row_num, _, cells in self.rows:
            yield row_num, dict(zip(self.columns, cells))


@dataclass
class Dataset:
    # Row tuples in COPY column order -- see load().
    places: list = field(default_factory=list)
    people: list = field(default_factory=list)
    events: list = field(default_factory=list)
    event_places: list = field(default_factory=list)
    event_route: list = field(default_factory=list)
    event_people: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    # Source rows dropped because they had no ID -- the places export carries
    # one ("malt"). Counted rather than silently skipped, so the CLI can say so.
    skipped: dict = field(default_factory=dict)


# ------------------------------------------------------------------ parsing --

def _cell(value) -> str:
    """A spreadsheet value as the text a person would see in the cell.

    openpyxl hands back numbers as numbers, and an ID typed into Excel can
    arrive as 354.0; the viewer should show 354.
    """
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _int(value):
    """A cell as an int ID, or None."""
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def _ids(value) -> list:
    """"354,951, 534" -> [354, 951, 534]. Raises on anything that is not an ID,
    so a leftover "TODO: ..." note cannot load as a silently shorter list."""
    out = []
    for part in str(value or "").split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"not an ID: {part!r}")
        out.append(int(part))
    return out


def _coord(lat, lng):
    """(lat, lng) as floats, or (None, None) when the source has no real point.

    The source writes "unknown" three ways -- blank, 0/0, and a guess with a
    question mark ("37?") -- and all three have to become NULL, because the map
    would otherwise plot them.
    """
    try:
        la, ln = float(str(lat).strip()), float(str(lng).strip())
    except ValueError:
        return None, None
    if la == 0 and ln == 0:
        return None, None
    if not (-90 <= la <= 90 and -180 <= ln <= 180):
        return None, None
    return la, ln


def _text(value) -> str:
    """A cell as clean text. The Notion exports write missing values as NULL."""
    text = str(value or "").strip()
    return "" if text == "NULL" else text


def _one(source_dir: Path, pattern: str) -> Path:
    matches = sorted(source_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {pattern!r} in {source_dir}, found {len(matches)}"
        )
    return matches[0]


def read_csv_source(key: str, path: Path, id_column: str) -> Source:
    # utf-8-sig: both Notion exports begin with a byte-order mark, which would
    # otherwise become part of the first column's name ("﻿PlaceID").
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        columns = next(reader)
        source = Source(key, path.name, columns)
        id_at = columns.index(id_column)
        for row_num, cells in enumerate(reader, start=2):
            cells = cells + [""] * (len(columns) - len(cells))
            source.rows.append((row_num, _int(cells[id_at]), cells))
    return source


def read_places(sources) -> tuple:
    rows, skipped = [], 0
    for source in sources:
        for _, r in source.records():
            place_id = _int(r.get("PlaceID"))
            if place_id is None:
                skipped += 1
                continue
            lat, lng = _coord(r.get("Lat", ""), r.get("Lng", ""))
            alt = ", ".join(
                _text(r.get(k)) for k in ("AltName", "AltSpelling") if _text(r.get(k))
            )
            rows.append((
                place_id, _text(r.get("PlaceName")), alt, lat, lng,
                _text(r.get("Comments")), _text(r.get("Verses")),
            ))
    return rows, skipped


def read_people(source: Source) -> tuple:
    rows, skipped = [], 0
    for _, r in source.records():
        person_id = _int(r.get("PersonID"))
        if person_id is None:
            skipped += 1
            continue
        rows.append((
            person_id, _text(r.get("Name")), _text(r.get("AltName")),
            _text(r.get("Descriptor")), _text(r.get("SubjectType")),
            _text(r.get("Gender")), _text(r.get("Verses")),
        ))
    return rows, skipped


def read_events(path: Path, ds: Dataset) -> Source:
    """The events workbook -> ds.events and the three link tables.

    event_id is the row's position among non-empty rows, which is Bible order.
    Imported here rather than at the top so the web app, which never reads a
    spreadsheet, does not need openpyxl importable.
    """
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = wb.active.iter_rows(values_only=True)
        columns = [_cell(h) for h in next(sheet)]
        raw = [[_cell(v) for v in row] for row in sheet]
    finally:
        wb.close()

    source = Source("events", path.name, columns)
    event_id = 0
    for row_num, cells in enumerate(raw, start=2):
        cells = (cells + [""] * len(columns))[:len(columns)]
        r = dict(zip(columns, cells))
        passage, title = _text(r.get("Passage (Logos Data)")), _text(r.get("Title"))
        if not passage and not title:
            source.rows.append((row_num, None, cells))  # a blank row, not an event
            continue
        event_id += 1
        source.rows.append((row_num, event_id, cells))
        ds.events.append((
            event_id, passage, title, _text(r.get("Description")), _text(r.get("Icon")),
        ))
        try:
            places = _ids(r.get("PlaceID"))
            route = _ids(r.get("RoutePlaceID"))
            people = _ids(r.get("resolved_people_ids"))
        except ValueError as err:
            raise ValueError(f"events row {row_num} ({passage}): {err}") from None
        # dict.fromkeys: de-duplicate, keeping first-seen order.
        for pos, pid in enumerate(dict.fromkeys(places)):
            ds.event_places.append((event_id, pid, pos))
        for pos, pid in enumerate(route):
            ds.event_route.append((event_id, pos, pid))
        for pos, pid in enumerate(dict.fromkeys(people)):
            ds.event_people.append((event_id, pid, pos))
    return source


def read_sources(source_dir) -> Dataset:
    """Every source file in `source_dir` -> one Dataset. Touches no database."""
    source_dir = Path(source_dir)
    ds = Dataset()
    places = [read_csv_source("places", _one(source_dir, PLACES_GLOB), "PlaceID")]
    if (source_dir / NEW_PLACES_FILE).exists():
        places.append(read_csv_source("new_places", source_dir / NEW_PLACES_FILE, "PlaceID"))
    people = read_csv_source("people", _one(source_dir, PEOPLE_GLOB), "PersonID")

    ds.places, ds.skipped["places"] = read_places(places)
    ds.people, ds.skipped["people"] = read_people(people)
    events = read_events(source_dir / EVENTS_FILE, ds)
    ds.sources = [events, *places, people]
    return ds


# --------------------------------------------------------------- validation --

def validate(ds: Dataset) -> list:
    """Every reason not to load `ds`, as readable lines. [] means load it.

    Collected up front rather than left to the foreign keys: Postgres stops at
    the first violation and names a constraint, where this names every bad ID
    and the event it is in, which is what someone fixing the spreadsheet needs.
    """
    problems = []
    for label, rows in (("place", ds.places), ("person", ds.people)):
        seen, dupes = set(), set()
        for r in rows:
            (dupes if r[0] in seen else seen).add(r[0])
        if dupes:
            problems.append(f"duplicate {label} IDs: {sorted(dupes)[:20]}")

    place_ids = {r[0] for r in ds.places}
    person_ids = {r[0] for r in ds.people}
    passage = {r[0]: r[1] for r in ds.events}
    checks = (
        ("place", ds.event_places, 1, place_ids),
        ("route place", ds.event_route, 2, place_ids),
        ("person", ds.event_people, 1, person_ids),
    )
    for label, links, idx, known in checks:
        for link in links:
            if link[idx] not in known:
                problems.append(
                    f"event {link[0]} ({passage[link[0]]}): unknown {label} ID {link[idx]}"
                )
    if not ds.events:
        problems.append("no events found")
    return problems


# --------------------------------------------------------------------- load --

def load(conn, ds: Dataset) -> dict:
    """Replace everything in the biblemap schema with `ds`. Returns row counts.

    Same shape as bible.load, for the same reasons: TRUNCATE-and-reload because
    the spreadsheets are the source of truth, COPY because the database may be
    across the network, one transaction so a failure leaves the old map intact.
    The verbatim source copy is replaced in the same transaction, so the viewer
    can never show a file other than the one the map was built from.
    """
    conn.execute(
        "TRUNCATE biblemap.event_places, biblemap.event_route, biblemap.event_people, "
        "biblemap.events, biblemap.people, biblemap.places, "
        "biblemap.source_rows, biblemap.source_files"
    )
    source_files = [(s.key, s.filename, json.dumps(s.columns), len(s.rows))
                    for s in ds.sources]
    source_rows = [(s.key, row_num, record_id, json.dumps(cells))
                   for s in ds.sources for row_num, record_id, cells in s.rows]
    tables = (
        ("places", "place_id, name, alt_names, lat, lng, comments, verses", ds.places),
        ("people", "person_id, name, alt_name, descriptor, subject_type, gender, verses",
         ds.people),
        ("events", "event_id, passage, title, description, icon", ds.events),
        ("event_places", "event_id, place_id, position", ds.event_places),
        ("event_route", "event_id, position, place_id", ds.event_route),
        ("event_people", "event_id, person_id, position", ds.event_people),
        ("source_files", "file_key, filename, columns, row_count", source_files),
        ("source_rows", "file_key, row_num, record_id, cells", source_rows),
    )
    with conn.cursor() as cur:
        for table, columns, rows in tables:
            with cur.copy(f"COPY biblemap.{table} ({columns}) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
    conn.commit()
    return {table: len(rows) for table, _, rows in tables}


# ------------------------------------------------------------------ reading --

def loaded(conn) -> bool:
    """Whether the schema exists AND holds events. See bible.loaded for why
    to_regclass rather than just querying."""
    if conn.execute("SELECT to_regclass('biblemap.events')").fetchone()[0] is None:
        return False
    return conn.execute("SELECT EXISTS (SELECT 1 FROM biblemap.events)").fetchone()[0]


def counts(conn) -> dict:
    row = conn.execute(
        """
        SELECT (SELECT count(*) FROM biblemap.events),
               (SELECT count(*) FROM biblemap.people),
               (SELECT count(*) FROM biblemap.places),
               (SELECT count(*) FROM biblemap.places WHERE lat IS NULL)
        """
    ).fetchone()
    return dict(zip(("events", "people", "places", "places_without_coords"), row))


def data(conn) -> dict:
    """Everything the page needs, in one payload.

    One call rather than an endpoint per view: the three views are filters over
    the same ~1,700 events, and filtering in the browser makes switching views
    instant. Places and people are limited to the ones some event mentions --
    the rest have nothing to show.
    """
    events = conn.execute(
        """
        SELECT e.event_id, e.passage, e.title, e.description, e.icon,
               COALESCE((SELECT array_agg(place_id ORDER BY position)
                         FROM biblemap.event_places p WHERE p.event_id = e.event_id), '{}'),
               COALESCE((SELECT array_agg(place_id ORDER BY position)
                         FROM biblemap.event_route r WHERE r.event_id = e.event_id), '{}'),
               COALESCE((SELECT array_agg(person_id ORDER BY position)
                         FROM biblemap.event_people p WHERE p.event_id = e.event_id), '{}')
        FROM biblemap.events e
        ORDER BY e.event_id
        """
    ).fetchall()
    places = conn.execute(
        """
        SELECT place_id, name, lat, lng
        FROM biblemap.places
        WHERE place_id IN (SELECT place_id FROM biblemap.event_places
                           UNION SELECT place_id FROM biblemap.event_route)
        ORDER BY name
        """
    ).fetchall()
    people = conn.execute(
        """
        SELECT person_id, name, descriptor
        FROM biblemap.people
        WHERE person_id IN (SELECT person_id FROM biblemap.event_people)
        ORDER BY name
        """
    ).fetchall()
    return {
        "events": [
            {"id": i, "passage": passage, "title": title, "description": desc,
             "icon": icon, "places": pl, "route": rt, "people": pp}
            for i, passage, title, desc, icon, pl, rt, pp in events
        ],
        "places": [
            {"id": i, "name": name, "lat": lat, "lng": lng}
            for i, name, lat, lng in places
        ],
        "people": [
            {"id": i, "name": name, "descriptor": descriptor}
            for i, name, descriptor in people
        ],
    }


# ----------------------------------------------------------- source viewer --

def source_files(conn) -> list:
    """The loaded source files, in load order: events, places, people."""
    return conn.execute(
        """
        SELECT file_key, filename, columns, row_count, loaded_at
        FROM biblemap.source_files
        ORDER BY array_position(ARRAY['events','places','new_places','people'], file_key)
        """
    ).fetchall()


def record_names(conn) -> dict:
    """{"place": {id: name}, "person": {id: name}} for every loaded record, so
    the viewer can label an ID link with who or what it points at."""
    return {
        "place": dict(conn.execute("SELECT place_id, name FROM biblemap.places").fetchall()),
        "person": dict(conn.execute("SELECT person_id, name FROM biblemap.people").fetchall()),
    }


def locate(conn, kind: str, record_id: int):
    """(file_key, row_num) of the source row a map record came from, or None."""
    keys = RECORD_SOURCES.get(kind)
    if not keys:
        return None
    return conn.execute(
        """
        SELECT file_key, row_num FROM biblemap.source_rows
        WHERE file_key = ANY(%s) AND record_id = %s
        ORDER BY array_position(%s, file_key)
        LIMIT 1
        """,
        (keys, record_id, keys),
    ).fetchone()


def source_page(conn, key: str, offset: int, limit: int, q: str = "") -> tuple:
    """One window of a source file's rows, optionally filtered by `q`.

    Returns (total_matching, rows) with rows as (row_num, record_id, cells).
    The filter is a literal, case-insensitive match against any cell -- the same
    escaping as bible.search, for the same reason.
    """
    where, params = "file_key = %s", [key]
    if q:
        pattern = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where += " AND cells::text ILIKE %s ESCAPE '\\'"
        params.append(f"%{pattern}%")
    total = conn.execute(
        f"SELECT count(*) FROM biblemap.source_rows WHERE {where}", params
    ).fetchone()[0]
    rows = conn.execute(
        f"""
        SELECT row_num, record_id, cells FROM biblemap.source_rows
        WHERE {where} ORDER BY row_num OFFSET %s LIMIT %s
        """,
        [*params, offset, limit],
    ).fetchall()
    return total, rows


def source_row_index(conn, key: str, row_num: int) -> int:
    """How many rows of `key` come before `row_num` -- the offset that puts it
    on screen."""
    return conn.execute(
        "SELECT count(*) FROM biblemap.source_rows WHERE file_key = %s AND row_num < %s",
        (key, row_num),
    ).fetchone()[0]
