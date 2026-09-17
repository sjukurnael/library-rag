"""The Bible map loader: spreadsheets -> Dataset -> biblemap schema -> page payload.

The source files are built per test in tmp_path, in the same layout and with
the same column names as the real export, so the parse is exercised on the
shapes that actually arrive -- a single ID stored as a number, "NULL" for
missing text, a "37?" coordinate, a row with no ID.
"""
import csv

import openpyxl
import pytest

from library_rag import biblemap

PLACE_COLS = ["PlaceID", "PlaceName", "AltName", "AltSpelling", "Comments", "Lat", "Lng", "Verses"]
PERSON_COLS = ["PersonID", "Name", "AltName", "Descriptor", "SubjectType", "Gender", "Verses"]
EVENT_COLS = ["Passage (Logos Data)", "Title", "Description", "Icon",
              "PlaceID", "RoutePlaceID", "resolved_people_ids"]


def _write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig, as Notion writes it: the BOM must not leak into "PlaceID".
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def _write_events(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(EVENT_COLS)
    for r in rows:
        ws.append(r)
    wb.save(path)


@pytest.fixture
def source(tmp_path):
    _write_csv(tmp_path / "Places-Data/Places Data abc123_all.csv", PLACE_COLS, [
        ["527", "Haran", "", "", "", "36.8638", "39.0321", "Gen 11:31"],
        ["1189", "Ur", "NULL", "", "", "30.9620", "46.1037", "Gen 11:28"],
        ["272", "Canaan", "Kenaanite", "", "region", "31.6935", "34.8438", ""],
        ["354", "Eden", "", "", "", "0", "0", ""],
        ["738", "Luz", "", "", "", "37?", "36.57?", ""],
        ["", "malt", "", "", "", "", "", ""],
    ])
    _write_csv(tmp_path / "Places-Data/New Places (added during resolution).csv",
               ["PlaceID", "PlaceName", "Lat", "Lng", "Comments"],
               [["10003", "Israel", "32.05", "35.25", "Region"]])
    _write_csv(tmp_path / "People Data/People Data xyz_all.csv", PERSON_COLS, [
        ["180", "Abraham", "Abram", "NULL", "Character", "M", "Gen 11:26"],
        ["1334", "Sarah", "Sarai", "", "Character", "F", ""],
    ])
    _write_events(tmp_path / biblemap.EVENTS_FILE, [
        ["Ge 2:8", "Eden", "Garden planted", "ph-tree", 354, None, "180"],
        [None, None, None, None, None, None, None],  # a blank row is not an event
        ["Ge 11:31", "Terah Moves to Haran", "", "ph-path",
         "1189,527,272,527", "1189, 527", "180, 1334"],
        ["Jdg 1:26", "Luz", "", "", "738", "527,272,527", 180],
    ])
    return tmp_path


def test_read_sources_parses_every_file(source):
    ds = biblemap.read_sources(source)
    assert [p[0] for p in ds.places] == [527, 1189, 272, 354, 738, 10003]
    assert ds.skipped == {"places": 1, "people": 0}
    assert [e[:3] for e in ds.events] == [
        (1, "Ge 2:8", "Eden"), (2, "Ge 11:31", "Terah Moves to Haran"), (3, "Jdg 1:26", "Luz")
    ], "event ids are Bible order, skipping the blank row"


def test_unknown_coordinates_become_none_not_zero(source):
    places = {p[0]: p for p in biblemap.read_sources(source).places}
    assert places[527][3:5] == (36.8638, 39.0321)
    assert places[354][3:5] == (None, None), "0/0 is 'unknown', not a point off Africa"
    assert places[738][3:5] == (None, None), "a '37?' guess is not a coordinate"


def test_null_text_is_empty(source):
    places = {p[0]: p for p in biblemap.read_sources(source).places}
    assert places[1189][2] == ""
    people = {p[0]: p for p in biblemap.read_sources(source).people}
    assert people[180][3] == ""


def test_place_links_are_a_set_but_routes_keep_every_stop(source):
    ds = biblemap.read_sources(source)
    assert [(p, pos) for e, p, pos in ds.event_places if e == 2] == [(1189, 0), (527, 1), (272, 2)]
    assert [p for e, _, p in ds.event_route if e == 3] == [527, 272, 527], \
        "a route that returns to its start keeps both visits"
    assert [p for e, p, _ in ds.event_people if e == 1] == [180], "a numeric cell is one ID"


def test_a_leftover_todo_note_refuses_to_parse(source):
    _write_events(source / biblemap.EVENTS_FILE, [
        ["Ge 11:31", "Terah", "", "", "1189,TODO: Harran not found in Places.xlsx", None, "180"],
    ])
    with pytest.raises(ValueError, match="Ge 11:31"):
        biblemap.read_sources(source)


def test_validate_names_every_unknown_id(source):
    ds = biblemap.read_sources(source)
    assert biblemap.validate(ds) == []
    ds.event_route.append((2, 9, 99999))
    ds.event_people.append((3, 424242, 1))
    problems = biblemap.validate(ds)
    assert len(problems) == 2
    assert "99999" in problems[0] and "Ge 11:31" in problems[0]
    assert "424242" in problems[1]


def test_load_and_read_back(conn, source):
    ds = biblemap.read_sources(source)
    assert biblemap.loaded(conn) is False
    written = biblemap.load(conn, ds)
    assert written["events"] == 3 and written["event_route"] == 5

    assert biblemap.loaded(conn) is True
    assert biblemap.counts(conn) == {
        "events": 3, "people": 2, "places": 6, "places_without_coords": 2,
    }

    d = biblemap.data(conn)
    terah = d["events"][1]
    assert terah["places"] == [1189, 527, 272]
    assert terah["route"] == [1189, 527]
    assert terah["people"] == [180, 1334]
    # Only places and people some event mentions -- Israel is loaded but unused.
    assert {p["id"] for p in d["places"]} == {527, 1189, 272, 354, 738}
    assert next(p for p in d["places"] if p["id"] == 354)["lat"] is None


def test_reloading_replaces_rather_than_appends(conn, source):
    ds = biblemap.read_sources(source)
    biblemap.load(conn, ds)
    biblemap.load(conn, ds)
    assert biblemap.counts(conn)["events"] == 3


# ------------------------------------------------------- the verbatim copy --

def test_every_source_row_is_kept_verbatim(source):
    """Including the rows the map drops, and the text the parse cleans up:
    the viewer exists to show what the file said, not what we made of it."""
    ds = biblemap.read_sources(source)
    files = {s.key: s for s in ds.sources}
    assert list(files) == ["events", "places", "new_places", "people"]

    places = files["places"]
    assert places.columns == PLACE_COLS
    ur = next(r for r in places.rows if r[1] == 1189)
    assert ur[0] == 3, "row numbers count the header as row 1, like Excel"
    assert ur[2][2] == "NULL", "the copy keeps NULL; only the parse blanks it"
    malt = next(r for r in places.rows if r[2][1] == "malt")
    assert malt[1] is None, "a row with no ID is kept, pointing at nothing"

    events = files["events"]
    assert [(r[0], r[1]) for r in events.rows] == [(2, 1), (3, None), (4, 2), (5, 3)]
    assert events.rows[0][2][4] == "354", "a numeric cell reads as the text in the cell"


def test_locate_and_page_through_the_copy(conn, source):
    biblemap.load(conn, biblemap.read_sources(source))

    assert biblemap.locate(conn, "event", 2) == ("events", 4)
    assert biblemap.locate(conn, "place", 10003) == ("new_places", 2)
    assert biblemap.locate(conn, "person", 1334) == ("people", 3)
    assert biblemap.locate(conn, "place", 999) is None

    keys = [f[0] for f in biblemap.source_files(conn)]
    assert keys == ["events", "places", "new_places", "people"]

    total, rows = biblemap.source_page(conn, "places", 0, 2)
    assert total == 6 and [r[0] for r in rows] == [2, 3]
    assert biblemap.source_row_index(conn, "places", 5) == 3

    total, rows = biblemap.source_page(conn, "places", 0, 50, q="haran")
    assert total == 1 and rows[0][1] == 527

    names = biblemap.record_names(conn)
    assert names["place"][527] == "Haran" and names["person"][1334] == "Sarah"


def test_every_link_column_is_a_real_events_column(source):
    """A renamed column would silently stop being clickable."""
    events = next(s for s in biblemap.read_sources(source).sources if s.key == "events")
    for column in biblemap.LINK_COLUMNS["events"]:
        if column != "Duplicate IDs":  # not in the minimal fixture
            assert column in events.columns, column
