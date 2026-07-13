"""Safe, allowlist-driven local action execution for GhostOS.

The public ``execute_action`` function accepts a canonical action name and a
plain argument mapping.  It never parses or executes shell text.  Every
handler validates its target through ``action_permissions`` and returns a
structured result instead of claiming success based on model output.
"""

from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path
from typing import Callable, Mapping

from action_permissions import (
    ActionPolicy,
    ActionValidationError,
    validate_application,
    validate_creation_path,
    validate_existing_path,
    validate_note,
    validate_url,
)
from action_registry import ActionDefinition, get_action, normalize_action_name


Result = dict[str, object]


def _result(
    success: bool,
    action: str,
    target: str | None,
    message: str,
    *,
    error_code: str | None = None,
) -> Result:
    result: Result = {
        "success": success,
        "action": action,
        "target": target,
        "message": message,
    }
    if error_code:
        result["error"] = {"code": error_code, "message": message}
    return result


class ActionAgent:
    """Execute only registered actions under one trusted local policy.

    OS-facing functions are injectable so tests can remain fully offline and
    never open a real application, browser, or Explorer window.
    """

    def __init__(
        self,
        policy: ActionPolicy | None = None,
        *,
        startfile: Callable[[str], object] | None = None,
        url_opener: Callable[..., object] | None = None,
        process_launcher: Callable[..., object] | None = None,
    ) -> None:
        self.policy = policy or ActionPolicy()
        self._startfile = startfile if startfile is not None else getattr(os, "startfile", None)
        self._url_opener = url_opener if url_opener is not None else webbrowser.open
        self._process_launcher = process_launcher if process_launcher is not None else subprocess.Popen

    def execute(self, action_name: object, arguments: Mapping[str, object] | None = None) -> Result:
        """Validate and execute one allowlisted action.

        This method does not infer actions from natural language.  The caller
        (normally the router) must choose a registry name and provide typed
        arguments; unknown or extra arguments are rejected.
        """

        normalized_name = normalize_action_name(action_name)
        definition = get_action(normalized_name)
        if definition is None:
            return _result(
                False,
                normalized_name or "unknown",
                None,
                "This action is not allowed.",
                error_code="action_not_allowed",
            )
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            return _result(
                False, definition.name, None, "Action arguments must be an object.",
                error_code="invalid_arguments",
            )

        argument_keys = set(arguments.keys())
        if not all(isinstance(key, str) for key in argument_keys):
            return _result(
                False, definition.name, None, "Action argument names must be strings.",
                error_code="invalid_arguments",
            )
        missing = set(definition.required_arguments) - argument_keys
        unexpected = argument_keys - definition.allowed_arguments
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing: " + ", ".join(sorted(missing)))
            if unexpected:
                details.append("unexpected: " + ", ".join(sorted(unexpected)))
            return _result(
                False,
                definition.name,
                self._raw_target(definition, arguments),
                "Invalid action arguments (" + "; ".join(details) + ").",
                error_code="invalid_arguments",
            )

        try:
            handler = getattr(self, f"_handle_{definition.name}")
            return handler(arguments)
        except ActionValidationError as exc:
            return _result(
                False,
                definition.name,
                self._raw_target(definition, arguments),
                exc.message,
                error_code=exc.code,
            )
        except FileExistsError:
            # Handles the atomic O_EXCL/mkdir race if another process creates
            # the target between validation and the filesystem operation.
            return _result(
                False,
                definition.name,
                self._raw_target(definition, arguments),
                "A file or folder already exists at this path.",
                error_code="already_exists",
            )
        except (OSError, RuntimeError) as exc:
            # Return a stable error without exposing a traceback or pretending
            # the operating system accepted the request.
            return _result(
                False,
                definition.name,
                self._raw_target(definition, arguments),
                f"The operating system could not complete this action: {exc}",
                error_code="action_failed",
            )

    @staticmethod
    def _raw_target(definition: ActionDefinition, arguments: Mapping[str, object]) -> str | None:
        if not definition.target_argument:
            return None
        target = arguments.get(definition.target_argument)
        return str(target) if target is not None else None

    def _require_startfile(self) -> Callable[[str], object]:
        if self._startfile is None:
            raise ActionValidationError(
                "unsupported_platform", "Opening local files and folders currently requires Windows."
            )
        return self._startfile

    def _launch_process(self, argv: list[str]) -> None:
        """Launch a fixed argv list and reject an immediate process failure."""

        process = self._process_launcher(argv, shell=False)
        if process is None:
            raise ActionValidationError("launch_failed", "Windows did not accept the application request.")
        poll = getattr(process, "poll", None)
        return_code = poll() if callable(poll) else getattr(process, "returncode", None)
        if return_code not in (None, 0):
            raise ActionValidationError(
                "launch_failed", f"The application exited immediately with code {return_code}."
            )

    def _handle_open_file(self, arguments: Mapping[str, object]) -> Result:
        path = validate_existing_path(arguments["path"], self.policy, expected="file")
        self._require_startfile()(str(path))
        return _result(True, "open_file", str(path), "Windows accepted the request to open this file.")

    def _handle_open_folder(self, arguments: Mapping[str, object]) -> Result:
        path = validate_existing_path(arguments["path"], self.policy, expected="folder")
        self._require_startfile()(str(path))
        return _result(True, "open_folder", str(path), "Windows accepted the request to open this folder.")

    def _handle_open_url(self, arguments: Mapping[str, object]) -> Result:
        url = validate_url(arguments["url"])
        accepted = self._url_opener(url, new=2)
        if accepted is not True:
            return _result(
                False, "open_url", url, "The default browser did not accept the URL.",
                error_code="launch_failed",
            )
        return _result(True, "open_url", url, "The default browser accepted the URL.")

    def _handle_open_app(self, arguments: Mapping[str, object]) -> Result:
        alias, executable = validate_application(arguments["app"], self.policy)
        # A list plus shell=False is mandatory: no user text is interpreted by
        # cmd.exe/PowerShell and this action has no per-request argument field.
        self._launch_process([str(executable)])
        return _result(True, "open_app", alias, f"Windows accepted the request to start {alias}.")

    def _handle_reveal_in_explorer(self, arguments: Mapping[str, object]) -> Result:
        if os.name != "nt":
            raise ActionValidationError(
                "unsupported_platform", "Reveal in File Explorer currently requires Windows."
            )
        path = validate_existing_path(arguments["path"], self.policy, expected="file")
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
        explorer = (windir / "explorer.exe").resolve(strict=False)
        if not explorer.is_file():
            raise ActionValidationError("application_unavailable", "Windows File Explorer is unavailable.")
        self._launch_process([str(explorer), "/select,", str(path)])
        return _result(
            True,
            "reveal_in_explorer",
            str(path),
            "Windows accepted the request to reveal this file in File Explorer.",
        )

    def _handle_create_text_note(self, arguments: Mapping[str, object]) -> Result:
        path, content = validate_note(arguments["path"], arguments["content"], self.policy)
        if path.exists():
            raise ActionValidationError("already_exists", "A file or folder already exists at this path.")
        if not path.parent.is_dir():
            raise ActionValidationError("parent_not_found", "The note's parent folder does not exist.")

        # O_EXCL makes the no-overwrite guarantee atomic, including if another
        # process creates the file after the validation check above.
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not path.is_file():
            raise RuntimeError("The note was not present after creation.")
        return _result(True, "create_text_note", str(path), "Text note created successfully.")

    def _handle_create_folder(self, arguments: Mapping[str, object]) -> Result:
        path = validate_creation_path(arguments["path"], self.policy)
        if path.exists():
            raise ActionValidationError("already_exists", "A file or folder already exists at this path.")
        if not path.parent.is_dir():
            raise ActionValidationError("parent_not_found", "The parent folder does not exist.")
        path.mkdir(parents=False, exist_ok=False)
        if not path.is_dir():
            raise RuntimeError("The folder was not present after creation.")
        return _result(True, "create_folder", str(path), "Folder created successfully.")


def execute_action(
    action_name: object,
    arguments: Mapping[str, object] | None = None,
    *,
    policy: ActionPolicy | None = None,
) -> Result:
    """Convenience integration API for the Flask route/router."""

    return ActionAgent(policy=policy).execute(action_name, arguments)


# Agent-shaped alias for code that names callable modules after their agent.
action_agent = execute_action
