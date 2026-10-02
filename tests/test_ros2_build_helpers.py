"""Portable tests for bounded build jobs and recoverable archive downloads."""
import hashlib
import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

HELPERS = Path(__file__).resolve().parents[1] / 'scripts/ros2_build_helpers.bash'


class RosBuildHelperTests(unittest.TestCase):
    def test_job_limit_reaches_make_and_cmake_and_rejects_zero(self):
        env = dict(os.environ, WMAL_ROS_BUILD_JOBS='3')
        result = subprocess.run(['bash', '-c', 'set -e; source "$1"; '
                                 'wmal_ros_build_jobs; printf "%s|%s" "$MAKEFLAGS" '
                                 '"$CMAKE_BUILD_PARALLEL_LEVEL"', 'bash', str(HELPERS)],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '-j3 -l3|3')
        env['WMAL_ROS_BUILD_JOBS'] = '0'
        failed = subprocess.run(['bash', '-c', 'source "$1"; wmal_ros_build_jobs',
                                 'bash', str(HELPERS)], env=env, capture_output=True)
        self.assertNotEqual(failed.returncode, 0)

    def test_corrupt_cache_is_preserved_and_replaced_by_verified_download(self):
        self.check_archive_recovery(failed_transfer=False)

    def test_interrupted_download_never_becomes_a_final_cache_entry(self):
        self.check_archive_recovery(failed_transfer=True)

    def check_archive_recovery(self, failed_transfer):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / 'fixture.tar.gz'
            with tarfile.open(fixture, 'w:gz') as archive:
                info = tarfile.TarInfo('repo-1.0/marker')
                info.size = 2
                archive.addfile(info, io.BytesIO(b'ok'))
            checksum = hashlib.sha256(fixture.read_bytes()).hexdigest()
            workspace = root / 'workspace'
            (workspace / 'downloads').mkdir(parents=True)
            (workspace / 'src').mkdir()
            target = workspace / 'downloads/repo-1.0.tar.gz'
            if not failed_transfer:
                target.write_bytes(b'corrupt')
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            curl = bin_dir / 'curl'
            # Stub only the external transfer; real hashing, extraction, cache and rename.
            curl.write_text('#!/usr/bin/env bash\ncp "$ROS_ARCHIVE_FIXTURE" "${@: -1}"\n'
                            + ('exit 22\n' if failed_transfer else ''))
            curl.chmod(0o755)
            env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}',
                       ROS_ARCHIVE_FIXTURE=str(fixture))
            command = ['bash', '-c', 'set -e; source "$1"; wmal_fetch_ros_source '
                       '"$2" repo 1.0 "$3"', 'bash', str(HELPERS), str(workspace), checksum]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if failed_transfer:
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(target.exists())
                curl.write_text('#!/usr/bin/env bash\ncp "$ROS_ARCHIVE_FIXTURE" "${@: -1}"\n')
                result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(target.read_bytes(), fixture.read_bytes())
            self.assertEqual((workspace / 'src/repo-1.0/marker').read_bytes(), b'ok')
            if not failed_transfer:
                self.assertEqual(len(list((workspace / 'downloads').glob('*.invalid.*'))), 1)


if __name__ == '__main__':
    unittest.main()
