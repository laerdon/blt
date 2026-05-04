#!/usr/bin/env bash
set -euo pipefail

ROOT=/alloc/blt
LOCKFILE=/tmp/tiny_decoder_plot.lock
SCRIPT="$ROOT/plot_data/tiny_decoder/make_tiny_decoder_vs_baseline_graph.py"

exec /usr/bin/flock -n "$LOCKFILE" /usr/bin/python3 "$SCRIPT"
