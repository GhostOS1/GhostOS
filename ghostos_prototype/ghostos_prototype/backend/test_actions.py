"""Offline tests for the GhostOS safe local action layer."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from action_agent import ActionAgent
from action_permissions import (
    ActionPolicy,
    ActionValidationError,
    ApplicationSpec,
    validate_application,
    validate_url,
)
from action_registry import ACTION_REGISTRY, get_action, list_actions


class _Process:
    returncode = None


class SafeActionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name).resolve()
        self.launches: list[tuple] = []
        self.app_executable = self.root / "safe-app.exe"
        self.app_executable.write_bytes(b"test executable placeholder")
        self.policy = ActionPolicy(
            user_home=self.root,
            safe_create_roots=(self.root,),
            open_roots=(self.root,),
            protected_roots=(),
            allowed_applications={"safe app": ApplicationSpec(self.app_executable)},
        )

        def fake_startfile(target):
            self.launches.append(("startfile", target))

        def fake_url(url, **kwargs):
            self.launches.append(("url", url, kwargs))
            return True

        def fake_process(argv, **kwargs):
            self.launches.append(("process", argv, kwargs))
            return _Process()

        self.agent = ActionAgent(
            self.policy,
            startfile=fake_startfile,
            url_opener=fake_url,
            process_launcher=fake_process,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_registry_is_an_explicit_allowlist(self):
        self.assertEqual(len(ACTION_REGISTRY), 7)
        self.assertIsNotNone(get_action("open-file"))
        self.assertIsNone(get_action("execute_shell"))
        self.assertNotIn("execute_shell", {item["name"] for item in list_actions()})

    def test_unknown_action_and_extra_arguments_are_rejected(self):
        unknown = self.agent.execute("run command", {"command": "whoami"})
        self.assertFalse(unknown["success"])
        self.assertEqual(unknown["error"]["code"], "action_not_allowed")

        target = self.root / "file.txt"
        target.write_text("hello", encoding="utf-8")
        extra = self.agent.execute("open_file", {"path": str(target), "args": ["--unsafe"]})
        self.assertFalse(extra["success"])
        self.assertEqual(extra["error"]["code"], "invalid_arguments")
        self.assertEqual(self.launches, [])

    def test_open_file_and_folder_use_validated_startfile(self):
        target = self.root / "file.txt"
        target.write_text("hello", encoding="utf-8")

        file_result = self.agent.execute("open_file", {"path": str(target)})
        folder_result = self.agent.execute("open_folder", {"path": str(self.root)})

        self.assertTrue(file_result["success"])
        self.assertTrue(folder_result["success"])
        self.assertEqual(self.launches[0], ("startfile", str(target.resolve())))
        self.assertEqual(self.launches[1], ("startfile", str(self.root)))

    def test_open_type_and_root_are_enforced(self):
        outside = self.root.parent / f"outside-{self.root.name}.txt"
        try:
            outside.write_text("outside", encoding="utf-8")
            wrong_type = self.agent.execute("open_file", {"path": str(self.root)})
            outside_result = self.agent.execute("open_file", {"path": str(outside)})
            self.assertEqual(wrong_type["error"]["code"], "invalid_target_type")
            self.assertEqual(outside_result["error"]["code"], "path_not_allowed")
            self.assertEqual(self.launches, [])
        finally:
            outside.unlink(missing_ok=True)

    def test_sensitive_paths_are_rejected(self):
        sensitive = self.root / ".ssh"
        sensitive.mkdir()
        key = sensitive / "id_rsa"
        key.write_text("private", encoding="utf-8")
        result = self.agent.execute("open_file", {"path": str(key)})
        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "sensitive_path")
        self.assertEqual(self.launches, [])

    def test_protected_root_is_rejected(self):
        protected = self.root / "protected"
        protected.mkdir()
        file_path = protected / "system.txt"
        file_path.write_text("system", encoding="utf-8")
        policy = ActionPolicy(
            user_home=self.root,
            safe_create_roots=(self.root,),
            open_roots=(self.root,),
            protected_roots=(protected,),
            allowed_applications={},
        )
        agent = ActionAgent(policy, startfile=lambda target: self.launches.append(target))
        result = agent.execute("open_file", {"path": str(file_path)})
        self.assertEqual(result["error"]["code"], "protected_path")
        self.assertEqual(self.launches, [])

    def test_urls_allow_http_https_and_reject_unsafe_schemes(self):
        accepted = self.agent.execute("open_url", {"url": "HTTPS://example.com/docs?q=ghost"})
        self.assertTrue(accepted["success"])
        self.assertEqual(accepted["target"], "https://example.com/docs?q=ghost")

        for unsafe in (
            "file:///C:/Windows/System32/config/SAM",
            "javascript:alert(1)",
            "https://user:password@example.com/",
            "https://example.com/a path",
            "https://example.com\\path",
            "https://example.com:bad/",
        ):
            with self.subTest(url=unsafe):
                with self.assertRaises(ActionValidationError):
                    validate_url(unsafe)

    def test_url_launch_failure_is_reported_honestly(self):
        agent = ActionAgent(self.policy, url_opener=lambda *_args, **_kwargs: False)
        result = agent.execute("open_url", {"url": "https://example.com"})
        self.assertFalse(result["success"])
        self.assertEqual(result["error"]["code"], "launch_failed")

    def test_application_alias_is_allowlisted_and_has_no_user_arguments(self):
        result = self.agent.execute("open_app", {"app": " SAFE   APP "})
        self.assertTrue(result["success"])
        self.assertEqual(result["target"], "safe app")
        kind, argv, kwargs = self.launches[-1]
        self.assertEqual(kind, "process")
        self.assertEqual(argv, [str(self.app_executable.resolve())])
        self.assertIs(kwargs["shell"], False)

        rejected = self.agent.execute("open_app", {"app": "powershell", "args": ["whoami"]})
        self.assertFalse(rejected["success"])
        self.assertEqual(rejected["error"]["code"], "invalid_arguments")

    def test_command_interpreter_is_rejected_even_if_configured(self):
        command = self.root / "cmd.exe"
        command.write_bytes(b"placeholder")
        policy = ActionPolicy(
            user_home=self.root,
            protected_roots=(),
            allowed_applications={"bad": ApplicationSpec(command)},
        )
        with self.assertRaises(ActionValidationError) as raised:
            validate_application("bad", policy)
        self.assertEqual(raised.exception.code, "application_not_allowed")

    def test_create_text_note_is_utf8_atomic_and_never_overwrites(self):
        note = self.root / "GhostOS note.md"
        first = self.agent.execute("create_text_note", {"path": str(note), "content": "नमस्ते GhostOS"})
        second = self.agent.execute("create_text_note", {"path": str(note), "content": "overwrite"})

        self.assertTrue(first["success"])
        self.assertEqual(note.read_text(encoding="utf-8"), "नमस्ते GhostOS")
        self.assertFalse(second["success"])
        self.assertEqual(second["error"]["code"], "already_exists")
        self.assertEqual(note.read_text(encoding="utf-8"), "नमस्ते GhostOS")

    def test_note_extension_content_and_parent_are_validated(self):
        bad_extension = self.agent.execute(
            "create_text_note", {"path": str(self.root / "note.exe"), "content": "no"}
        )
        bad_content = self.agent.execute(
            "create_text_note", {"path": str(self.root / "note.txt"), "content": b"bytes"}
        )
        missing_parent = self.agent.execute(
            "create_text_note", {"path": str(self.root / "missing" / "note.txt"), "content": "no"}
        )
        self.assertEqual(bad_extension["error"]["code"], "invalid_note_type")
        self.assertEqual(bad_content["error"]["code"], "invalid_content")
        self.assertEqual(missing_parent["error"]["code"], "parent_not_found")

    def test_create_folder_and_traversal_outside_safe_root(self):
        folder = self.root / "New Project"
        created = self.agent.execute("create_folder", {"path": str(folder)})
        duplicate = self.agent.execute("create_folder", {"path": str(folder)})
        escaped = self.agent.execute(
            "create_folder", {"path": str(self.root / ".." / f"escaped-{self.root.name}")}
        )

        self.assertTrue(created["success"])
        self.assertTrue(folder.is_dir())
        self.assertEqual(duplicate["error"]["code"], "already_exists")
        self.assertEqual(escaped["error"]["code"], "path_not_allowed")

    @unittest.skipUnless(os.name == "nt", "File Explorer is a Windows action")
    def test_reveal_uses_argument_list_without_shell(self):
        target = self.root / "file.txt"
        target.write_text("hello", encoding="utf-8")
        fake_windows = self.root / "windows"
        fake_windows.mkdir()
        (fake_windows / "explorer.exe").write_bytes(b"placeholder")
        with patch.dict(os.environ, {"WINDIR": str(fake_windows)}):
            result = self.agent.execute("reveal_in_explorer", {"path": str(target)})
        self.assertTrue(result["success"])
        kind, argv, kwargs = self.launches[-1]
        self.assertEqual(kind, "process")
        self.assertEqual(argv[1:], ["/select,", str(target.resolve())])
        self.assertIs(kwargs["shell"], False)


if __name__ == "__main__":
    unittest.main()
