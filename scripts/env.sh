# Source this, do not run it:   source scripts/env.sh
#
# The three paths the pipeline needs that are NOT in git, because they are 92 GB
# and because a dataset does not belong in a repository. They live outside any
# checkout on purpose: two checkouts existed at once during the vrgrid-26
# migration, and holding the data inside one of them meant deleting that
# checkout would have destroyed the dataset, the FRNet checkpoints and every
# figure behind a published number.
#
# Override VRGRID_ASSETS for a machine that stores them elsewhere -- the AWS
# instance will:
#     VRGRID_ASSETS=/mnt/data source scripts/env.sh

: "${VRGRID_ASSETS:=$HOME/Desktop/sih26/assets}"

export VRGRID_ASSETS
export VRGRID_DATA_ROOT="$VRGRID_ASSETS/dataset"
export VRGRID_FRNET_CHECKPOINT="$VRGRID_ASSETS/checkpoints/frnet-semantickitti_seg.pth"

# assets/
#   dataset/            90 GB   SemanticKITTI, sequences 00-21 + poses
#   checkpoints/       423 MB   FRNet .pth, including the fine-tuned ones
#   figures/            50 MB   the PNG/SVG/CSV behind published figures
#   rerun-recordings/  1.3 GB   ORPHANED -- baked .rrd files from the removed
#                               Rerun dashboard. Kept only until someone decides
#                               they are not wanted; nothing reads them now.

for _v in VRGRID_DATA_ROOT VRGRID_FRNET_CHECKPOINT; do
  eval "_p=\$$_v"
  [ -e "$_p" ] || echo "warning: $_v -> $_p does not exist" >&2
done
unset _v _p
