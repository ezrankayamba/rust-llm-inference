#!/bin/sh
# Turns RLI_* variables into rli-server flags. Arguments given to `docker run` are appended,
# so any flag can still be passed directly.
set -eu

fetch() {
    # -C - resumes the .part an interrupted start left behind. curl drops a header given with
    # -H on a redirect to another host, so a token stays with RLI_MODEL_URL's host.
    curl -fL --no-progress-meter --retry 5 -C - -o "$part" "$@" "$RLI_MODEL_URL"
}

download() {
    if [ -n "${HF_TOKEN:-}" ]; then
        # From stdin, so the token isn't in curl's command line.
        printf 'Authorization: Bearer %s\n' "$HF_TOKEN" | fetch -H @-
    else
        fetch
    fi
}

if [ ! -f "$RLI_MODEL" ]; then
    if [ -z "${RLI_MODEL_URL:-}" ]; then
        echo "rli: no model at RLI_MODEL=$RLI_MODEL; mount its directory (e.g. -v /srv/models:/models)" \
            "or set RLI_MODEL_URL to download it" >&2
        exit 1
    fi
    dir=$(dirname "$RLI_MODEL")
    if [ ! -w "$dir" ]; then
        echo "rli: can't download into $dir: it must be writable by uid $(id -u)" >&2
        exit 1
    fi
    part="$RLI_MODEL.part"
    echo "rli: downloading $RLI_MODEL_URL to $RLI_MODEL" >&2
    download || {
        rc=$?
        # 33: the server ignores ranges, so a .part can't be resumed and would fail every start.
        [ "$rc" -eq 33 ] || exit "$rc"
        echo "rli: the server can't resume $part; downloading it again" >&2
        rm -f "$part"
        download
    }
    if [ -n "${RLI_MODEL_SHA256:-}" ] &&
        ! echo "$RLI_MODEL_SHA256  $part" | sha256sum -c --status; then
        rm -f "$part"
        echo "rli: $RLI_MODEL_URL does not match RLI_MODEL_SHA256; removed the download" >&2
        exit 1
    fi
    # Renamed only when whole, so a later start never serves part of a file.
    mv "$part" "$RLI_MODEL"
    echo "rli: downloaded $RLI_MODEL" >&2
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
