#!/bin/sh
# Turns RLI_* variables into rli-server flags. Arguments given to `docker run` are appended,
# so any flag can still be passed directly.
set -eu

if [ ! -f "$RLI_MODEL" ]; then
    echo "rli: no model at RLI_MODEL=$RLI_MODEL; mount its directory (e.g. -v /srv/models:/models:ro)" >&2
    exit 1
fi

set -- --model "$RLI_MODEL" --host "$RLI_HOST" --port "$RLI_PORT" "$@"
if [ -n "${RLI_CTX:-}" ]; then set -- "$@" --ctx "$RLI_CTX"; fi
if [ -n "${RLI_GPU_LAYERS:-}" ]; then set -- "$@" --gpu-layers "$RLI_GPU_LAYERS"; fi
if [ -n "${RLI_THREADS:-}" ]; then set -- "$@" --threads "$RLI_THREADS"; fi
if [ -n "${RLI_DRAFT:-}" ]; then set -- "$@" --draft "$RLI_DRAFT"; fi
if [ -n "${RLI_DRAFT_TOKENS:-}" ]; then set -- "$@" --draft-tokens "$RLI_DRAFT_TOKENS"; fi
# Bare switches such as "--repack --coarse-acts"; split on spaces on purpose.
# shellcheck disable=SC2086
if [ -n "${RLI_EXTRA_ARGS:-}" ]; then set -- "$@" $RLI_EXTRA_ARGS; fi

exec rli-server "$@"
