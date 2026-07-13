import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import connect_system


class BackgroundDiscoveryExclusionTests(unittest.TestCase):
    def test_discovered_root_inside_configured_exclusion_is_not_watched_or_indexed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            allowed_root = temp_root / "allowed"
            excluded_root = temp_root / "private"
            discovered_project = excluded_root / "secret-project"
            allowed_root.mkdir()
            discovered_project.mkdir(parents=True)

            cancel_was_set = connect_system._bg_cancel_event.is_set()
            connect_system._bg_cancel_event.clear()
            try:
                with (
                    patch.object(
                        connect_system,
                        "discover_vscode_projects",
                        return_value=[discovered_project],
                    ),
                    patch.object(
                        connect_system,
                        "discover_git_repositories",
                        return_value=[],
                    ),
                    patch.object(connect_system, "watch_folders") as watch_folders,
                    patch.object(
                        connect_system,
                        "index_folders",
                        return_value={"stats": {}, "cancelled": False},
                    ) as index_folders,
                    patch.dict(connect_system._bg_state, dict(connect_system._bg_state), clear=True),
                ):
                    connect_system._run_background_discovery_and_indexing(
                        standard_folders=[allowed_root],
                        include_dev_folders=True,
                        scan_entire_drives=False,
                        excluded_folders=[excluded_root],
                    )

                    self.assertEqual(
                        watch_folders.call_args.args[0],
                        [str(allowed_root)],
                    )
                    self.assertEqual(
                        index_folders.call_args.args[0],
                        [str(allowed_root)],
                    )
            finally:
                if cancel_was_set:
                    connect_system._bg_cancel_event.set()
                else:
                    connect_system._bg_cancel_event.clear()

    def test_background_status_reset_is_idle_and_refuses_active_work(self):
        original = dict(connect_system._bg_state)
        cancel_was_set = connect_system._bg_cancel_event.is_set()
        try:
            with connect_system._bg_lock:
                connect_system._bg_state.update({
                    "active": False,
                    "done": True,
                    "phase": "finished",
                    "files_completed": 42,
                    "failed_files": [{"path": "old.txt"}],
                    "stats": {"total_files": 42},
                })
            connect_system._bg_cancel_event.set()

            self.assertTrue(connect_system.reset_background_status())
            status = connect_system.get_background_status()
            self.assertEqual(status, connect_system._idle_background_state())
            self.assertFalse(connect_system._bg_cancel_event.is_set())

            with connect_system._bg_lock:
                connect_system._bg_state.update({
                    "active": True,
                    "done": False,
                    "phase": "indexing",
                    "files_completed": 7,
                })
            self.assertFalse(connect_system.reset_background_status())
            self.assertEqual(connect_system.get_background_status()["files_completed"], 7)
        finally:
            with connect_system._bg_lock:
                connect_system._bg_state.clear()
                connect_system._bg_state.update(original)
            if cancel_was_set:
                connect_system._bg_cancel_event.set()
            else:
                connect_system._bg_cancel_event.clear()


if __name__ == "__main__":
    unittest.main()
