#!/usr/bin/env bash
# Build Python 3.10 bindings against the installed Jazzy C++ middleware.
set -eo pipefail
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: conda activate wmal; bash scripts/build_ros2_wmal.bash'
    echo 'Requires Ubuntu 24.04, ROS Jazzy, colcon, CMake, g++, pybind11-dev.'
    exit 0
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != wmal || -z "${CONDA_PREFIX:-}" ]]; then
    echo 'Activate the project environment first: conda activate wmal' >&2
    exit 1
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/scripts/ros2_build_helpers.bash"
project_python="$CONDA_PREFIX/bin/python"
"$project_python" -c 'import sys; assert sys.version_info[:2] == (3, 10)'
if [[ ! -f /opt/ros/jazzy/setup.bash || ! -x /usr/bin/colcon ]]; then
    echo 'Install system ROS Jazzy and python3-colcon-common-extensions first.' >&2
    exit 1
fi
source /opt/ros/jazzy/setup.bash
set -u
# Underlay libraries must match the source API/ABI we are overriding.
"$project_python" - <<'PY'
import xml.etree.ElementTree as ET
from pathlib import Path
versions = {'rclpy': '7.1.12', 'rcl_interfaces': '2.0.4',
            'common_interfaces': '5.3.8', 'unique_identifier_msgs': '2.5.1'}
for package, expected in versions.items():
    actual = ET.parse(Path('/opt/ros/jazzy/share') / package / 'package.xml').findtext('version')
    if actual != expected:
        raise SystemExit(f'{package} is {actual}, expected {expected}; review source pins before rebuilding.')
PY
workspace="$project_root/ros2/.python310"
mkdir -p "$workspace/src" "$workspace/downloads"
"$project_python" -m pip install -r "$project_root/requirements-ros2-build.txt"

# Versions match this machine's Jazzy binary packages; do not silently use rolling.
wmal_fetch_ros_source "$workspace" rclpy 7.1.12 4d4bfd63bdedb6386e1ee705817fe818095dd53f8b9147512de3d0ea76bbf551
wmal_fetch_ros_source "$workspace" rcl_interfaces 2.0.4 297c3d227f4f349afb65302291be308d2f4a648c9e45c24c71fd60af2c1e1c73
wmal_fetch_ros_source "$workspace" common_interfaces 5.3.8 aa38d6fd3232909c858173ff60ccb9aeb4d7d3aa6867a65cfcd1a0e2679b87ff
wmal_fetch_ros_source "$workspace" unique_identifier_msgs 2.5.1 5be66480a8c38285ffb64aa0c895c5408f7df72f3ac5170c5a383d5f6ad0b3c6

packages=(builtin_interfaces rcl_interfaces rosgraph_msgs action_msgs service_msgs
    type_description_interfaces lifecycle_msgs unique_identifier_msgs
    std_msgs geometry_msgs sensor_msgs rclpy wmal_interfaces)
export PATH="$CONDA_PREFIX/bin:$PATH"
# Limit compiler memory consumption independently from colcon's package executor.
wmal_ros_build_jobs
/usr/bin/colcon --log-base "$workspace/log" build \
    --base-paths "$workspace/src" "$project_root/ros2/wmal_interfaces" \
    --build-base "$workspace/build" --install-base "$workspace/install" \
    --merge-install --executor sequential --packages-select "${packages[@]}" \
    --packages-ignore test_msgs --allow-overriding "${packages[@]:0:12}" \
    --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
    "-DPython3_EXECUTABLE=$project_python" \
    "-DPYTHON_EXECUTABLE=$project_python" "-DPython3_ROOT_DIR=$CONDA_PREFIX" \
    -DPython3_FIND_STRATEGY=LOCATION
set +u
source "$project_root/scripts/setup_ros2.bash"
echo 'ROS bindings ready. In each project terminal: source scripts/setup_ros2.bash'
