# Helpers shared by the local build entry point and portable behavioral tests.
wmal_ros_build_jobs() {
    local jobs="${WMAL_ROS_BUILD_JOBS:-2}"
    if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then
        echo 'WMAL_ROS_BUILD_JOBS must be a positive integer.' >&2
        return 1
    fi
    export MAKEFLAGS="-j$jobs -l$jobs"
    export CMAKE_BUILD_PARALLEL_LEVEL="$jobs"
}

wmal_fetch_ros_source() {
    local workspace="$1" repository="$2" version="$3" checksum="$4"
    local archive partial backup
    archive="$workspace/downloads/$repository-$version.tar.gz"
    if [[ ! -f "$archive" ]] || ! printf '%s  %s\n' "$checksum" "$archive" | sha256sum --check --status -; then
        if [[ -f "$archive" ]]; then
            backup="$(mktemp "$archive.invalid.XXXXXX")" || return 1
            mv "$archive" "$backup" || return 1
            echo "Preserved invalid download: $backup" >&2
        fi
        partial="$(mktemp "$workspace/downloads/.$repository-$version.XXXXXX.part")" || return 1
        if ! curl --fail --location --retry 2 --connect-timeout 15 --max-time 120 \
                "https://codeload.github.com/ros2/$repository/tar.gz/refs/tags/$version" -o "$partial"; then
            echo "Download failed; partial file kept at $partial; rerun to retry." >&2
            return 1
        fi
        printf '%s  %s\n' "$checksum" "$partial" | sha256sum --check - || return 1
        mv "$partial" "$archive" || return 1
    fi
    if [[ ! -d "$workspace/src/$repository-$version" ]]; then
        tar -xzf "$archive" -C "$workspace/src" || return 1
    fi
}
