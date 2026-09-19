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
    assert [e[:4] for e in ds.events] == [
        (1, 1, "Ge 2:8", "Eden"),
        (2, 2, "Ge 11:31", "Terah Moves to Haran"),
        (3, 3, "Jdg 1:26", "Luz"),
    ], "on a first import id and seq agree; the blank row is neither"


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


# ----------------------------------------------------------------- editing --
#
# The stored rows are the data and the map is derived from them, so what these
# pin is that the two can never disagree: every edit re-parses everything, and
# one that would not parse leaves nothing behind.

def _loaded(conn, source):
    biblemap.load(conn, biblemap.read_sources(source))
    return conn


def test_rebuilding_from_the_stored_rows_matches_the_files(conn, source):
    _loaded(conn, source)
    before = biblemap.data(conn)
    biblemap.rebuild(conn)
    conn.commit()
    assert biblemap.data(conn) == before


def test_an_edit_changes_the_map(conn, source):
    _loaded(conn, source)
    row = biblemap.locate(conn, "place", 354)[1]
    biblemap.update_row(conn, "places", row, {"Lat": "31.0", "Lng": "47.4"},
                        editor="me@example.com")
    eden = next(p for p in biblemap.data(conn)["places"] if p["id"] == 354)
    assert (eden["lat"], eden["lng"]) == (31.0, 47.4)
    assert biblemap.edit_count(conn) == 1
    assert conn.execute(
        "SELECT editor, column_name, old_value, new_value FROM biblemap.source_edits "
        "WHERE column_name = 'Lat'"
    ).fetchone() == ("me@example.com", "Lat", "0", "31.0")


def test_an_edit_that_would_break_the_map_is_refused_whole(conn, source):
    _loaded(conn, source)
    before = biblemap.data(conn)
    row = biblemap.locate(conn, "event", 2)[1]
    with pytest.raises(biblemap.Invalid) as err:
        biblemap.update_row(conn, "events", row, {"PlaceID": "1189,99999"})
    assert "unknown place ID 99999" in err.value.problems[0]
    # Neither the map nor the row it came from moved.
    assert biblemap.data(conn) == before
    assert biblemap.edit_count(conn) == 0


def test_inserting_a_row_shifts_the_rows_below_and_keeps_event_ids(conn, source):
    _loaded(conn, source)
    ids_before = {e["passage"]: e["id"] for e in biblemap.data(conn)["events"]}

    out = biblemap.insert_row(conn, "events", 2, editor="me@example.com")
    assert out["row"] == 3
    # A blank row is not an event, so nothing on the map changed yet...
    assert {e["passage"]: e["id"] for e in biblemap.data(conn)["events"]} == ids_before
    # ...and the rows below moved down, as they would in a spreadsheet.
    assert biblemap.locate(conn, "event", 2) == ("events", 5)

    biblemap.update_row(conn, "events", 3, {"Passage (Logos Data)": "Ge 3:1",
                                            "Title": "Inserted", "PlaceID": "527"})
    events = biblemap.data(conn)["events"]
    new = next(e for e in events if e["title"] == "Inserted")
    assert [e["title"] for e in events][:2] == ["Eden", "Inserted"], "it lands in place"
    assert new["id"] not in ids_before.values(), "a new event takes a new id"
    assert {e["passage"]: e["id"] for e in events if e["passage"] in ids_before} == ids_before, \
        "every existing event keeps the id links point at"


def test_a_new_place_row_arrives_with_a_free_id(conn, source):
    _loaded(conn, source)
    out = biblemap.insert_row(conn, "places", 2)
    assert out["cells"][0] == "10004", "continues the 10000+ range, past Israel at 10003"
    biblemap.update_row(conn, "places", out["row"], {"PlaceName": "Somewhere",
                                                     "Lat": "31.5", "Lng": "35.0"})
    # Referencing it from an event is now legal, which is the whole point.
    biblemap.update_row(conn, "events", 4, {"PlaceID": "10004"})
    assert [p["name"] for p in biblemap.data(conn)["places"] if p["id"] == 10004] == ["Somewhere"]


def test_deleting_a_row_removes_it_and_records_what_it_was(conn, source):
    _loaded(conn, source)
    row = biblemap.locate(conn, "event", 3)[1]
    biblemap.delete_row(conn, "events", row, editor="me@example.com")
    assert [e["id"] for e in biblemap.data(conn)["events"]] == [1, 2]
    snapshot = conn.execute(
        "SELECT row_snapshot FROM biblemap.source_edits WHERE action = 'delete'"
    ).fetchone()[0]
    assert snapshot[1] == "Luz"


def test_export_round_trips(conn, source, tmp_path):
    _loaded(conn, source)
    biblemap.update_row(conn, "places", 2, {"PlaceName": "Haran (edited)"})

    out = tmp_path / "export"
    biblemap.export(conn, out)
    again = biblemap.read_sources(out)
    assert [p[1] for p in again.places if p[0] == 527] == ["Haran (edited)"]
    # ...and a fresh load of the export is the same map.
    before = biblemap.data(conn)
    biblemap.load(conn, again)
    assert biblemap.data(conn) == before
