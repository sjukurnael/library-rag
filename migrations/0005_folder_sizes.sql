-- Folders get a size. Drive never reports one -- size_bytes is NULL on every
-- folder row -- so the browser's Size column showed a dash for exactly the rows
-- where a size is most useful. The mirror already holds everything needed to
-- compute it: every PDF's size_bytes and the full parent_id tree.
--
-- Stored, not computed per request, because the answer only changes when the
-- mirror changes. mirror.materialise_sizes recomputes it after every sync, the
-- same arrangement as `path`. Note the total covers what the mirror covers:
-- PDFs. Other Drive content is never listed, so it never counts.

ALTER TABLE drive_files ADD COLUMN subtree_bytes BIGINT;

-- Existing rows: compute in place rather than forcing a Drive re-sync. Same
-- statement mirror.materialise_sizes runs -- pair every folder with each of its
-- descendants, then sum the descendants' file sizes per folder. The anchor row
-- pairs a folder with itself, so an empty folder still lands in the aggregate
-- and comes out 0, never NULL.
WITH RECURSIVE anc AS (
    SELECT f.file_id AS folder_id, f.file_id AS node_id
    FROM drive_files f
    WHERE f.mime_type = 'application/vnd.google-apps.folder'
  UNION ALL
    SELECT a.folder_id, d.file_id
    FROM drive_files d JOIN anc a ON d.parent_id = a.node_id
)
UPDATE drive_files f
SET subtree_bytes = s.total
FROM (
    SELECT anc.folder_id, COALESCE(SUM(d.size_bytes), 0) AS total
    FROM anc JOIN drive_files d ON d.file_id = anc.node_id
    GROUP BY anc.folder_id
) s
WHERE f.file_id = s.folder_id;
