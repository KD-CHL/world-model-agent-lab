# Source after `conda activate wmal`. Do not source this in system ROS terminals.
_wmal_setup_ros2() {
    local project_root overlay
    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    overlay="$project_root/ros2/.python310/install"
    if [[ "${CONDA_DEFAULT_ENV:-}" != wmal || -z "${CONDA_PREFIX:-}" ]]; then
        echo 'Activate the project environment first: conda activate wmal' >&2
        return 1
    fi
    if ! "$CONDA_PREFIX/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 10)' ; then
        echo 'This ROS overlay requires the project Python 3.10 environment.' >&2
        return 1
    fi
    if [[ ! -f "$overlay/local_setup.bash" || ! -f /opt/ros/jazzy/setup.bash ]]; then
        echo 'ROS overlay missing. Run: bash scripts/build_ros2_wmal.bash' >&2
        return 1
    fi
    source /opt/ros/jazzy/setup.bash || return 1
    source "$overlay/local_setup.bash" || return 1
    export PYTHONPATH="$project_root/src:${PYTHONPATH:-}"
    # Opt-in, host-local simulation; preserve user domain/RMW but never broaden discovery.
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-73}"
    export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
    "$CONDA_PREFIX/bin/python" -c 'import rclpy; from wmal_interfaces.srv import G1Session; from rosidl_generator_py import import_type_support; import_type_support("wmal_interfaces")' || {
        echo 'ROS Python bindings did not load; rebuild this overlay in wmal.' >&2
        return 1
    }
}

if _wmal_setup_ros2; then
    unset -f _wmal_setup_ros2
    return 0
else
    unset -f _wmal_setup_ros2
    return 1
fi
