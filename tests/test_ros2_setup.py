"""Exercise the project's opt-in ROS shell setup, including the actual ABI."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'scripts/setup_ros2.bash'


class RosSetupTests(unittest.TestCase):
    def test_rejects_non_project_environment_without_changing_ros_domain(self):
        env = dict(os.environ, CONDA_DEFAULT_ENV='not-wmal', ROS_DOMAIN_ID='91')
        result = subprocess.run(['bash', '--noprofile', '--norc', '-c',
                                 'if source "$1"; then exit 3; fi; '
                                 'test "$ROS_DOMAIN_ID" = 91', 'bash', str(SETUP)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(Path('/opt/ros/jazzy').is_dir()
                         and Path(sys.prefix).name == 'wmal'
                         and (ROOT / 'ros2/.python310/install/local_setup.bash').is_file()
                         and (ROOT / 'ros2/.python310/install/lib/python3.10/site-packages/'
                              'wmal_interfaces').is_dir(),
                         'Requires ROS Jazzy, wmal and the locally built project overlay')
    def test_setup_loads_matching_bindings_and_generated_session_interface(self):
        env = dict(os.environ, CONDA_DEFAULT_ENV='wmal', CONDA_PREFIX=sys.prefix,
                   ROS_AUTOMATIC_DISCOVERY_RANGE='SUBNET')
        result = subprocess.run(['bash', '--noprofile', '--norc', '-c',
                                 'set -e; source "$1"; "$2" -c '
                                 '\'import os, rclpy; from wmal_interfaces.srv import G1Session; '
                                 'from rosidl_generator_py import import_type_support; '
                                 'import_type_support("wmal_interfaces"); '
                                 'import rclpy.impl.implementation_singleton as i; '
                                 'assert "cpython-310" in i.rclpy_implementation.__file__; '
                                 'assert os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] == "LOCALHOST"; '
                                 'assert G1Session.Request(json="hello").json == "hello"\'',
                                 'bash', str(SETUP), sys.executable],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
