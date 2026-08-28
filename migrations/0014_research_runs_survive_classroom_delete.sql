-- Deleting a classroom that has ever been used was a 500.
--
-- 0012 added research_runs.classroom_id as a bare REFERENCES, which defaults to
-- ON DELETE NO ACTION. So the moment a reader asked the tutor one question, the
-- classroom became undeletable:
--
--   ForeignKeyViolation: update or delete on table "classrooms" violates
--   foreign key constraint "research_runs_classroom_id_fkey" on table
--   "research_runs"
--
-- Found by deleting a classroom through the HTTP surface after asking it two
-- questions. The unit test for delete passed throughout, because it deletes a
-- classroom that was never used -- the only kind that worked.
--
-- SET NULL rather than CASCADE, deliberately. research_runs holds the answer
-- text, and that is the reason it is durable at all; cascading would make
-- "delete this shelf" quietly destroy every answer ever given from it. This
-- keeps the same rule the rest of the schema follows -- deleting a classroom
-- removes the shelf, never the things that were on it or came off it. The run
-- survives as an unfiled answer, which is exactly what it now is.

ALTER TABLE research_runs
    DROP CONSTRAINT research_runs_classroom_id_fkey,
    ADD  CONSTRAINT research_runs_classroom_id_fkey
         FOREIGN KEY (classroom_id) REFERENCES classrooms(id) ON DELETE SET NULL;
