#!/usr/bin/env bash
#
# Deploy the ingestion worker as a Cloud Run JOB, triggered on demand by the
# app, with a daily sweep as a backstop.
#
# Why a Job and not the web container: Cloud Run SERVICES allocate CPU per
# request. `background.add_task(_drain_queue, ...)` starts after the response is
# flushed, so the instance is throttled to near-zero CPU for the entire time the
# book is actually being processed -- and is free to be evicted mid-book. The
# claim/lease in db.claim_next_book makes that safe (a stale claim is reaped
# after CLAIM_STALE_MINUTES) but not RELIABLE: the book just sits in
# 'processing' with no error until someone triggers another drain. Jobs get full
# un-throttled CPU for their whole run and are not tied to a request at all.
#
# WHO TRIGGERS IT. Two things, and the cron is the lesser one:
#
#   1. The app, the moment it queues rows (jobs.run_ingest_job). This is the
#      normal path, and it is why indexing now starts in seconds rather than
#      waiting on a clock.
#   2. This daily sweep, covering the one case the app cannot: nothing
#      re-drains unless somebody queues something, so a book left half-processed
#      by a killed execution would otherwise wait for the next click. Most days
#      the sweep finds an empty queue and exits in about a second.
#
#      It also clears a poison book -- one that reliably kills its instance, so
#      process_book's `except` never runs and it is never marked failed. Each
#      claim still increments `attempts` (claim_next_book does it in the same
#      UPDATE), so MAX_ATTEMPTS claims later it is marked failed and stops
#      blocking the front of the queue. Unattended that takes three days here;
#      by hand, three clicks.
#
# Costing, against the 180,000 vCPU-second / 360,000 GiB-second monthly free
# tier (aggregated per BILLING ACCOUNT, not per project):
#
#   Cloud Run jobs bill instance-based, for the entire lifetime of the instance,
#   with a MINIMUM OF 1 MINUTE per execution -- an idle drain finishes in about
#   a second and still bills 60s. That minimum is why the sweep is daily rather
#   than hourly: on-demand executions already cover the real work.
#
#     daily sweep:  30 execs x 60s x 2 vCPU  =   3,600 vCPU-s
#     on demand:   ~50 execs x 60s x 2 vCPU  =   6,000 vCPU-s
#     + the existing web service             =  13,800 vCPU-s  (measured, README)
#                                               -------------
#                                               ~23,400 of 180,000  -> 13%
#
#   leaving ~157,000 vCPU-s, about 21 hours a month of real 2-vCPU indexing,
#   still free. For reference an HOURLY sweep would be 87,600 vCPU-s on its own,
#   and every 30 MINUTES would be 175,200 -- over the allowance once the web
#   service is added. Redo this arithmetic before tightening the schedule.
#
# Usage:  ./scripts/deploy_ingest_job.sh
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project)}"
REGION="${REGION:-us-west1}"
JOB="${JOB:-library-rag-ingest}"
SCHEDULE="${SCHEDULE:-0 4 * * *}"          # 04:00 daily. See costing above.
SA="${SA:-library-rag-scheduler}"
SA_EMAIL="${SA}@${PROJECT}.iam.gserviceaccount.com"
SERVICE="${SERVICE:-library-rag}"          # the web service that triggers on demand

echo "project=${PROJECT} region=${REGION} job=${JOB}"

# --- 1. the job itself -------------------------------------------------------
#
# --source . reuses the SAME Dockerfile the web service is built from, so there
# is one image definition and no chance of the worker drifting from the app.
# --command/--args override the Dockerfile's uvicorn CMD; with no ENTRYPOINT set
# this resolves to `python -m library_rag.cli.ingest`, which is now a pure drain:
# no discovery, and no Drive client unless a Drive book is actually claimed.
#
# --max-retries=1 because the queue already owns retry semantics (MAX_ATTEMPTS
# plus the stale-claim reaper). Letting Cloud Run retry as well would mean two
# mechanisms racing over the same attempts budget.
#
# 2Gi because /tmp is tmpfs -- the downloaded PDF, the extracted markdown and the
# manifest all count against instance memory (see LIBRARY_RAG_DATA_DIR in the
# Dockerfile). 1Gi is what makes large scans OOM today.
gcloud run jobs deploy "${JOB}" \
  --source . \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --command python \
  --args=-m,library_rag.cli.ingest \
  --cpu=2 \
  --memory=2Gi \
  --task-timeout=3600 \
  --max-retries=1 \
  --env-vars-file=env.yaml

# --- 2. an identity for the scheduler ---------------------------------------
# Cloud Scheduler calls the Run Admin API as this account. It needs run.invoker
# on the JOB specifically -- not project-wide -- so a compromised scheduler
# cannot invoke anything else.
if ! gcloud iam service-accounts describe "${SA_EMAIL}" --project "${PROJECT}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SA}" \
    --project "${PROJECT}" \
    --display-name "Triggers the library-rag ingest job"
fi

gcloud run jobs add-iam-policy-binding "${JOB}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --member "serviceAccount:${SA_EMAIL}" \
  --role roles/run.invoker

# --- 3. let the WEB SERVICE trigger it on demand ------------------------------
# This is the path that actually matters: jobs.run_ingest_job() POSTs to the Run
# Admin API as the web service's own identity, so that identity needs the same
# job-scoped run.invoker. Read the account off the deployed service rather than
# assuming -- an unset serviceAccountName means the default compute account.
WEB_SA="$(gcloud run services describe "${SERVICE}" \
  --project "${PROJECT}" --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)')"
if [ -z "${WEB_SA}" ]; then
  PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
  WEB_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi
echo "web service runs as: ${WEB_SA}"

gcloud run jobs add-iam-policy-binding "${JOB}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --member "serviceAccount:${WEB_SA}" \
  --role roles/run.invoker

# INGEST_JOB_NAME is what flips the app from draining in-process to delegating.
# It has to live in env.yaml, not just be patched on here: the service is
# deployed with --env-vars-file, which REPLACES the whole set, so a variable
# only ever set by `services update` disappears at the next deploy and the app
# silently reverts to the throttled in-process drain.
if ! grep -q '^INGEST_JOB_NAME:' env.yaml 2>/dev/null; then
  echo
  echo "  !! Add this line to env.yaml, or the next --env-vars-file deploy"
  echo "  !! will silently revert the app to draining in-process:"
  echo
  echo "        INGEST_JOB_NAME: \"${JOB}\""
  echo
fi

gcloud run services update "${SERVICE}" \
  --project "${PROJECT}" \
  --region "${REGION}" \
  --update-env-vars "INGEST_JOB_NAME=${JOB},INGEST_JOB_REGION=${REGION}"

# --- 4. the daily backstop ----------------------------------------------------
# Idempotent: create on first run, update on every run after.
URI="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT}/jobs/${JOB}:run"

if gcloud scheduler jobs describe "${JOB}-daily" --location "${REGION}" --project "${PROJECT}" >/dev/null 2>&1; then
  VERB=update
else
  VERB=create
fi

gcloud scheduler jobs "${VERB}" http "${JOB}-daily" \
  --project "${PROJECT}" \
  --location "${REGION}" \
  --schedule "${SCHEDULE}" \
  --uri "${URI}" \
  --http-method POST \
  --oauth-service-account-email "${SA_EMAIL}"

echo
echo "Done. Run it once now with:"
echo "  gcloud run jobs execute ${JOB} --region ${REGION} --project ${PROJECT} --wait"
echo "Logs:"
echo "  gcloud run jobs executions list --job ${JOB} --region ${REGION} --project ${PROJECT}"
