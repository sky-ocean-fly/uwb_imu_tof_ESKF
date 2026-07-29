#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage:" >&2
  echo "  $0 --config COLLECTION_YAML --metadata SESSION_YAML --session-id ID [--output-root DIR]" >&2
  echo "  $0 --session-dir EXISTING_SESSION_DIR" >&2
}

collection_config=""
metadata_file=""
session_id=""
output_root=""
existing_session_dir=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      collection_config="$2"
      shift 2
      ;;
    --metadata)
      metadata_file="$2"
      shift 2
      ;;
    --session-id)
      session_id="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --session-dir)
      existing_session_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

package_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${package_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "$existing_session_dir" ]]; then
  if [[ -n "$collection_config" || -n "$metadata_file" || -n "$session_id" || -n "$output_root" ]]; then
    echo "--session-dir cannot be combined with session creation arguments" >&2
    usage
    exit 2
  fi
  session_json="$(python3 -c '
import json, pathlib, sys
directory = pathlib.Path(sys.argv[1]).expanduser().resolve()
manifest_path = directory / "session_manifest.json"
if not manifest_path.is_file():
    raise SystemExit("missing session manifest: %s" % manifest_path)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required = ("uwb", "mocap_a", "mocap_b", "status")
topics = manifest.get("topics", {})
missing = [name for name in required if not topics.get(name)]
if missing:
    raise SystemExit("session manifest has unresolved topics: %s" % ", ".join(missing))
prefix = directory / "raw_topics"
existing = sorted(prefix.parent.glob(prefix.name + "*.bag*"))
if existing:
    raise SystemExit("refusing to overwrite/reuse existing bag output: %s" % existing[0])
print(json.dumps({"session_dir": str(directory), "bag_prefix": str(prefix), "topics": topics}))
' "$existing_session_dir")"
else
  if [[ -z "$collection_config" || -z "$metadata_file" || -z "$session_id" ]]; then
    usage
    exit 2
  fi
  prepare_args=(
    -m uwb_bias_learning.session prepare
    --collection-config "$collection_config"
    --metadata "$metadata_file"
    --session-id "$session_id"
  )
  if [[ -n "$output_root" ]]; then
    prepare_args+=(--output-root "$output_root")
  fi
  session_json="$(python3 "${prepare_args[@]}")"
fi
session_dir="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["session_dir"])' "$session_json")"
bag_prefix="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["bag_prefix"])' "$session_json")"
mapfile -t topics < <(python3 -c 'import json,sys; print("\n".join(json.loads(sys.argv[1])["topics"].values()))' "$session_json")

echo "Session directory: ${session_dir}"
echo "Bag prefix: ${bag_prefix}"
echo "Topics: ${topics[*]}"

python3 -m uwb_bias_learning.session event --session-dir "$session_dir" --name RECORDING_STARTED

bag_pid=""
finish() {
  local signal_name="$1"
  if [[ -n "$bag_pid" ]] && kill -0 "$bag_pid" 2>/dev/null; then
    kill -INT "$bag_pid" 2>/dev/null || true
    wait "$bag_pid" || true
  fi
  python3 -m uwb_bias_learning.session event \
    --session-dir "$session_dir" --name "RECORDING_STOPPED_${signal_name}" || true
}
trap 'finish INT; exit 130' INT
trap 'finish TERM; exit 143' TERM

rosbag record --output-name "$bag_prefix" "${topics[@]}" &
bag_pid=$!
set +e
wait "$bag_pid"
bag_status=$?
set -e
bag_pid=""

python3 -m uwb_bias_learning.session event \
  --session-dir "$session_dir" --name "RECORDING_EXIT_${bag_status}"
exit "$bag_status"
