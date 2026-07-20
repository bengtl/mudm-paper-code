#!/usr/bin/env bash
# Launch the FlyWire second-connectome economy (the MANC replication).
#
# RUN THIS ON THE BOX (oden), from a mudm-tools git checkout, with this repo's
# `mudm_paper` package importable (pip install -e <repo>  OR  PYTHONPATH=<repo>).
#
# Two stages so you never burn GPU hours on a bad subsample:
#   STAGE A (cheap): download the stratified subsample + build labels, then STOP
#            and print the per-target class-survival breakdown for you to eyeball.
#   STAGE B (expensive): tile the pyramid + run the DGCNN sweep. Only runs when
#            you re-invoke with STAGE=B.
#
# Usage:
#   # stage A (download + prep + breakdown):
#   MUDM_TOOLS=/path/to/mudm-tools bash run_flywire_economy.sh
#   # ...eyeball the breakdown, then stage B (tile + sweep, long; run under tmux/nohup):
#   STAGE=B MUDM_TOOLS=/path/to/mudm-tools bash run_flywire_economy.sh
set -euo pipefail

# --- config (override via env) ---
N=${N:-30000}                                   # per-superclass stratified subsample cap
DATA=${DATA:-/data/ai/mudm-paper/flywire}       # meshes + labels + pyramid live here (on /data)
OUT=${OUT:-$PWD/results}                         # where flywire_economy_*.json are written
export TMPDIR=${TMPDIR:-/data/ai/mudm-paper/tmp} # box root is small — keep tiling temp on /data
MUDM_TOOLS=${MUDM_TOOLS:-$PWD}                    # dir containing scripts/download_flywire.py
STAGE=${STAGE:-A}

ANNOT="$DATA/Supplemental_file1_neuron_annotations.tsv"
MESHES="$DATA/meshes"
META="$DATA/flywire_meta.json"
PYR="$DATA/pyramid"

mkdir -p "$DATA" "$TMPDIR" "$OUT"

# --- preflight ---
python -c "import mudm_paper" 2>/dev/null || {
  echo "ERROR: 'mudm_paper' not importable. pip install -e <repo>  or  export PYTHONPATH=<repo>." >&2; exit 1; }
[[ -f "$MUDM_TOOLS/scripts/download_flywire.py" ]] || {
  echo "ERROR: $MUDM_TOOLS/scripts/download_flywire.py not found. Set MUDM_TOOLS=<mudm-tools checkout>." >&2; exit 1; }

echo "== FlyWire economy (stage $STAGE) :: N=$N DATA=$DATA OUT=$OUT TMPDIR=$TMPDIR =="

if [[ "$STAGE" == "A" ]]; then
  python -c "import cloudvolume" 2>/dev/null || {
    echo "ERROR: cloud-volume not available in this env (needed for the mesh download)." >&2; exit 1; }
  echo "[A1] downloading stratified subsample (per-superclass, cap $N) ..."
  python "$MUDM_TOOLS/scripts/download_flywire.py" --download \
      --select per-superclass --max-neurons "$N" --lod 1 --data-dir "$DATA"
  echo "[A2] mapping annotations -> labels + class-survival breakdown ..."
  python -m mudm_paper.flywire_prep --annotations "$ANNOT" --obj-dir "$MESHES" --out "$META"
  echo
  echo ">>> GATE: review the breakdown above. If cell_class AND hemilineage clear enough"
  echo ">>> classes at >=20 members, proceed to the expensive stage (run under tmux/nohup):"
  echo ">>>   STAGE=B N=$N DATA=$DATA OUT=$OUT MUDM_TOOLS=$MUDM_TOOLS bash $0"
  exit 0
fi

# --- STAGE B: tile + sweep (long, GPU) ---
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "(warning: nvidia-smi not found — sweep expects GPUs)"
[[ -f "$META" ]] || { echo "ERROR: $META missing — run STAGE A first." >&2; exit 1; }

echo "[B1] tiling f32 QEM pyramid -> $PYR ..."
python -m mudm_paper.tile_parquet --obj-dir "$MESHES" --meta "$META" --out "$PYR" --max-zoom 4

echo "[B2] economy sweep (DGCNN; z0-z4; B=512; 15 seeds; split grouped by cell_type) ..."
python -m mudm_paper.run_manc --parquet "$PYR" --out-dir "$OUT" \
    --labels super_class,cell_class,hemilineage --group-field cell_type --prefix flywire_economy

echo "[done] wrote $OUT/flywire_economy_*.json"
echo "Render:  python figures/make_table1.py --results $OUT --prefix flywire_economy"
