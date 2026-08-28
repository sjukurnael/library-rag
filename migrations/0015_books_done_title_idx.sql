-- The queue page took ten seconds to load.
--
-- /api/books listed every indexed book with its passage count, and got the
-- count by joining chunks and grouping:
--
--     SELECT b.id, ..., count(c.id) FROM books b JOIN chunks c ON c.book_id = b.id
--     WHERE b.status = 'done' GROUP BY b.id ORDER BY b.title
--
-- That aggregates all 1,758,032 chunk rows to produce 8,679 numbers, on every
-- page load, and the page cannot render until it finishes. Measured at 9.3 s.
--
-- The fix is to page the list and count per row instead, but ORDER BY title had
-- no index, so every page still had to sort the whole table first -- offset
-- 8,000 measured 9.8 s, no better than before.
--
-- Partial, because the list this serves is only ever the done ones: 912 kB
-- rather than an index over every book in every state. (title, id) rather than
-- (title) so the sort is fully ordered and pagination cannot repeat or skip a
-- row when two books share a title -- and 189 of these do, the same book filed
-- twice under different names.
--
--   first page   9,300 ms -> 70 ms
--   offset 8000  9,800 ms -> 1,526 ms

CREATE INDEX IF NOT EXISTS books_done_title_idx
    ON books (title, id) WHERE status = 'done';
