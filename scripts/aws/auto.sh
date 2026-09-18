#!/usr/bin/env bash
# The whole approved T4 pass, unattended, stopping at the first failure. [Shrestha]
#
#   scripts/aws/auto.sh labelled
#
# 1. wait (every 10 min, up to 24 h) until EC2 and S3 answer on this account
# 2. preflight + an EC2 dry-run launch -- both free; a refusal ends it here,
#    before a byte is uploaded
# 3. stage the chosen data to S3, then launch, setup, run (which fetches the
#    results and STOPS the instance)
# Every spending guardrail lives in t4.sh; this only sequences it.
set -euo pipefail
which="${1:?labelled or all}"
HERE="$(cd "$(dirname "$0")" && pwd)"
AWS="${AWS:-$HOME/.local/bin/aws}"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

for i in $(seq 1 144); do
    if "$AWS" --region ap-south-1 ec2 describe-availability-zones > /dev/null 2>&1 \
       && "$AWS" --region ap-south-1 s3api list-buckets > /dev/null 2>&1; then
        log "EC2 and S3 active"; break
    fi
    [[ $i == 144 ]] && { log "not active after 24 h -- contact AWS support"; exit 2; }
    sleep 600
done

log "preflight";  "$HERE/t4.sh" preflight
log "dry run";    "$HERE/t4.sh" dryrun
log "stage $which"; "$HERE/t4.sh" stage "$which"
log "launch";     "$HERE/t4.sh" launch
log "setup";      "$HERE/t4.sh" setup
log "run";        "$HERE/t4.sh" run
log "done -- results in docs/gpu-lane/t4, instance stopped"
"$HERE/t4.sh" status
