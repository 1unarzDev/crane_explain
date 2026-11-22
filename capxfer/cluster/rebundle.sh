#!/usr/bin/env bash
# Ship the current capxfer code to the research cluster. Prints the commit sha that jobs must name (repos[].ref).
#   cluster/rebundle.sh ["commit message"]
# 1. rsync ~/crane_explain/capxfer -> the capxfer-cluster worktree (no .venv/output/__pycache__), commit there;
# 2. make a capxfer-only commit (tree = that commit's capxfer/, parent = the previous one) on capxfer-cluster-pkg,
#    so bundles hold ~2 MB of capxfer, not crane_explain's history (Windows hosts also choke on its long paths);
# 3. research bundle push (once per commit). Local commits only: NEVER git push either branch anywhere.
set -euo pipefail
SRC=${CAPXFER_SRC:-$HOME/crane_explain/capxfer}
WT=${CAPXFER_WT:-$HOME/worktrees/capxfer-cluster}
PROJECT=${RC_PROJECT:-modular}
MSG=${1:-"capxfer: sync for research-cluster bundle"}
exec 9>"$HOME/.cache/capxfer-rebundle.lock"; flock 9          # one agent at a time
[ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = capxfer-cluster ] || { echo "worktree is not on capxfer-cluster" >&2; exit 1; }
rsync -a --delete --exclude .venv --exclude output --exclude __pycache__ --exclude .pytest_cache "$SRC/" "$WT/capxfer/"
git -C "$WT" add -A capxfer
git -C "$WT" diff --cached --quiet || git -C "$WT" commit -q -m "$MSG"
TREE=$(git -C "$WT" rev-parse HEAD:capxfer)
PREV=$(git -C "$WT" rev-parse -q --verify refs/heads/capxfer-cluster-pkg || true)
if [ -n "$PREV" ] && [ "$(git -C "$WT" rev-parse "$PREV^{tree}")" = "$TREE" ]; then
  PKG=$PREV
else
  PKG=$(git -C "$WT" commit-tree "$TREE" ${PREV:+-p "$PREV"} -m "$MSG (capxfer/ only; from capxfer-cluster $(git -C "$WT" rev-parse --short HEAD))")
  git -C "$WT" update-ref refs/heads/capxfer-cluster-pkg "$PKG"
fi
"$HOME/research-cluster/bin/research" bundle push -p "$PROJECT" capxfer "$WT" "$PKG" >&2
echo "$PKG"
