"""Validation and permission policy for GhostOS local actions.

All validation happens before an operating-system API is called.  Opening an
existing path is denied for known sensitive/system locations; optional
``open_roots`` can make that policy even narrower.  Creation is deliberately
stricter and is confined to the user's home directory (or roots supplied by
trusted local configuration).
"""

from __future__ import annotations

import os
import re
import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


MAX_URL_LENGTH = 4096
MAX_NOTE_BYTES = 1_000_000
NOTE_EXTENSIONS = frozenset({".txt", ".md"})

# Match path components, not the complete path string.  This avoids blocking a
# benign directory such as "my-ssh-presentation" while still protecting the
# real credential/cache names below.
SENSITIVE_COMPONENTS = frozenset({
    ".ssh", ".gnupg", ".aws", ".azure", ".kube",
    "1password", "bitwarden", "keepass", "credentials", "secrets",
    "login data", "cookies", "wallet.dat", "keychain", "user data",
    "credential manager",
})
SENSITIVE_PREFIXES = (".env", "id_rsa", "id_ed25519", "private_key")

# Even an allowlist entry may not point at a command interpreter or a common
# Windows script/proxy host.  This is defense in depth against unsafe local
# configuration and preserves the "no arbitrary shell" invariant.
PROHIBITED_EXECUTABLES = frozenset({
    "cmd.exe", "command.com", "powershell.exe", "pwsh.exe",
    "wscript.exe", "cscript.exe", "mshta.exe", "rundll32.exe",
    "regsvr32.exe", "wmic.exe",
})
PROHIBITED_EXECUTABLE_SUFFIXES = frozenset({
    ".bat", ".cmd", ".ps1", ".vbs", ".vbe", ".js", ".jse", ".wsf", ".hta",
})


class ActionValidationError(ValueError):
    """A safe, user-displayable validation failure with a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class ApplicationSpec:
    """Trusted configuration for one launchable application alias.

    No per-request command-line arguments are supported.  An application is
    therefore selected by alias and cannot be turned into a shell command by
    the model or API caller.
    """

    executable: Path


def _default_protected_roots() -> tuple[Path, ...]:
    candidates: list[Path] = []
    if os.name == "nt":
        for variable in (
            "WINDIR", "SystemRoot", "ProgramFiles", "ProgramW6432",
            "ProgramFiles(x86)", "ProgramData",
        ):
            value = os.environ.get(variable)
            if value:
                candidates.append(Path(value))
    else:
        candidates.extend(Path(path) for path in (
            "/bin", "/boot", "/dev", "/etc", "/proc", "/root", "/sbin",
            "/sys", "/usr", "/var",
        ))
    return tuple(candidates)


def discover_default_applications() -> Mapping[str, ApplicationSpec]:
    """Discover a conservative Windows application allowlist.

    Fixed installation locations are checked rather than ``PATH`` so a file
    with a familiar name in the working directory cannot be launched.  A
    caller may provide a different trusted mapping through ``ActionPolicy``.
    """

    if os.name != "nt":
        return MappingProxyType({})

    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    local_app_data = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))

    candidates: dict[str, tuple[Path, ...]] = {
        "notepad": (windir / "System32" / "notepad.exe",),
        "calculator": (windir / "System32" / "calc.exe",),
        "paint": (windir / "System32" / "mspaint.exe",),
        "vscode": (
            local_app_data / "Programs" / "Microsoft VS Code" / "Code.exe",
            program_files / "Microsoft VS Code" / "Code.exe",
        ),
        "chrome": (
            program_files / "Google" / "Chrome" / "Application" / "chrome.exe",
            program_files_x86 / "Google" / "Chrome" / "Application" / "chrome.exe",
        ),
        "edge": (
            program_files_x86 / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            program_files / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        ),
        "firefox": (
            program_files / "Mozilla Firefox" / "firefox.exe",
            program_files_x86 / "Mozilla Firefox" / "firefox.exe",
        ),
    }

    discovered: dict[str, ApplicationSpec] = {}
    for alias, paths in candidates.items():
        executable = next((path for path in paths if path.is_file()), None)
        if executable is not None:
            discovered[alias] = ApplicationSpec(executable=executable)
    return MappingProxyType(discovered)


@dataclass(frozen=True, slots=True)
class ActionPolicy:
    """Trusted local permission configuration used for every action."""

    user_home: Path = field(default_factory=Path.home)
    safe_create_roots: tuple[Path, ...] | None = None
    # ``None`` means existing paths may be opened anywhere except protected
    # locations.  Supplying roots confines opens/reveals to those roots.
    open_roots: tuple[Path, ...] | None = None
    protected_roots: tuple[Path, ...] = field(default_factory=_default_protected_roots)
    allowed_applications: Mapping[str, ApplicationSpec] = field(
        default_factory=discover_default_applications
    )

    def __post_init__(self) -> None:
        home = _canonical_root(self.user_home)
        create_roots = self.safe_create_roots
        if create_roots is None:
            create_roots = (home,)

        object.__setattr__(self, "user_home", home)
        object.__setattr__(self, "safe_create_roots", tuple(_canonical_root(p) for p in create_roots))
        if self.open_roots is not None:
            object.__setattr__(self, "open_roots", tuple(_canonical_root(p) for p in self.open_roots))
        object.__setattr__(self, "protected_roots", tuple(_canonical_root(p) for p in self.protected_roots))

        normalized_apps: dict[str, ApplicationSpec] = {}
        for alias, spec in self.allowed_applications.items():
            normalized_alias = _normalize_app_alias(alias)
            if normalized_alias:
                normalized_apps[normalized_alias] = spec
        object.__setattr__(self, "allowed_applications", MappingProxyType(normalized_apps))


def _canonical_root(path: os.PathLike | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_within(path: Path, root: Path) -> bool:
    """Case-aware containment without unsafe string-prefix matching."""

    try:
        path_value = os.path.normcase(str(path))
        root_value = os.path.normcase(str(root))
        return os.path.commonpath((path_value, root_value)) == root_value
    except (ValueError, OSError):
        # Different Windows drives, malformed paths, etc. are not contained.
        return False


def _contains_sensitive_component(path: Path) -> bool:
    for component in path.parts:
        lowered = component.casefold()
        if lowered in SENSITIVE_COMPONENTS:
            return True
        if any(lowered == prefix or lowered.startswith(prefix + ".") for prefix in SENSITIVE_PREFIXES):
            return True
    return False


def _validate_common_path(raw_path: object, policy: ActionPolicy, *, must_exist: bool) -> Path:
    if not isinstance(raw_path, (str, os.PathLike)):
        raise ActionValidationError("invalid_path", "Path must be a string.")
    value = os.fspath(raw_path).strip()
    if not value:
        raise ActionValidationError("invalid_path", "Path cannot be empty.")
    if "\x00" in value or any(ord(char) < 32 for char in value):
        raise ActionValidationError("invalid_path", "Path contains invalid control characters.")

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ActionValidationError("path_not_absolute", "An absolute path is required.")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        raise ActionValidationError("path_not_found", "The requested path does not exist.") from exc
    except (OSError, RuntimeError) as exc:
        raise ActionValidationError("invalid_path", "The requested path could not be resolved safely.") from exc

    if _contains_sensitive_component(resolved):
        raise ActionValidationError("sensitive_path", "Access to this sensitive path is not allowed.")
    if any(_is_within(resolved, root) for root in policy.protected_roots):
        raise ActionValidationError("protected_path", "Access to this protected system path is not allowed.")
    return resolved


def validate_existing_path(
    raw_path: object,
    policy: ActionPolicy,
    *,
    expected: str | None = None,
) -> Path:
    """Validate an existing file/folder for open or reveal actions."""

    path = _validate_common_path(raw_path, policy, must_exist=True)
    if policy.open_roots is not None and not any(_is_within(path, root) for root in policy.open_roots):
        raise ActionValidationError("path_not_allowed", "This path is outside the allowed open locations.")
    if expected == "file" and not path.is_file():
        raise ActionValidationError("invalid_target_type", "The requested path is not a file.")
    if expected == "folder" and not path.is_dir():
        raise ActionValidationError("invalid_target_type", "The requested path is not a folder.")
    return path


def validate_creation_path(raw_path: object, policy: ActionPolicy) -> Path:
    """Validate a not-yet-created path against creation roots and traversal."""

    path = _validate_common_path(raw_path, policy, must_exist=False)
    if not any(_is_within(path, root) for root in policy.safe_create_roots or ()):
        raise ActionValidationError(
            "path_not_allowed",
            "New files and folders may only be created inside an approved local folder.",
        )

    # Windows ignores trailing dots/spaces and supports alternate data streams
    # after a colon.  Reject those spellings to keep validation and creation
    # referring to exactly the same object.
    name = path.name
    if not name or name.endswith((".", " ")) or ":" in name:
        raise ActionValidationError("invalid_path", "The requested name is not safe on Windows.")
    if os.name == "nt" and Path(name).is_reserved():
        raise ActionValidationError("invalid_path", "The requested name is reserved by Windows.")
    return path


def validate_note(path_value: object, content: object, policy: ActionPolicy) -> tuple[Path, bytes]:
    """Validate a text-note target and return its UTF-8 bytes."""

    path = validate_creation_path(path_value, policy)
    if path.suffix.casefold() not in NOTE_EXTENSIONS:
        raise ActionValidationError("invalid_note_type", "Text notes must use a .txt or .md extension.")
    if not isinstance(content, str):
        raise ActionValidationError("invalid_content", "Note content must be text.")
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_NOTE_BYTES:
        raise ActionValidationError("note_too_large", "Text notes are limited to 1 MB.")
    return path, encoded


def validate_url(raw_url: object) -> str:
    """Accept only well-formed HTTP(S) URLs without embedded credentials."""

    if not isinstance(raw_url, str):
        raise ActionValidationError("invalid_url", "URL must be a string.")
    url = raw_url.strip()
    if not url or len(url) > MAX_URL_LENGTH:
        raise ActionValidationError("invalid_url", "URL is empty or too long.")
    if "\\" in url or any(char.isspace() or ord(char) < 32 for char in url):
        raise ActionValidationError("invalid_url", "URL contains invalid whitespace or control characters.")

    try:
        parsed = urlsplit(url)
        # Accessing .port performs additional validation and can raise.
        _ = parsed.port
    except ValueError as exc:
        raise ActionValidationError("invalid_url", "URL is malformed.") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise ActionValidationError("invalid_url", "Only HTTP and HTTPS URLs with a host are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise ActionValidationError("invalid_url", "URLs containing credentials are not allowed.")

    hostname = parsed.hostname or ""
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        if hostname.casefold() != "localhost":
            try:
                ascii_hostname = hostname.encode("idna").decode("ascii")
            except UnicodeError as exc:
                raise ActionValidationError("invalid_url", "URL host is malformed.") from exc
            labels = ascii_hostname.rstrip(".").split(".")
            if (
                not labels
                or any(
                    not label
                    or len(label) > 63
                    or label.startswith("-")
                    or label.endswith("-")
                    or not re.fullmatch(r"[A-Za-z0-9-]+", label)
                    for label in labels
                )
            ):
                raise ActionValidationError("invalid_url", "URL host is malformed.")

    # Normalize the scheme only; preserve the user's path/query/fragment.
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc, parsed.path, parsed.query, parsed.fragment))


_APP_ALIAS_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._ -]{0,63}$")


def _normalize_app_alias(alias: object) -> str:
    if not isinstance(alias, str):
        return ""
    normalized = " ".join(alias.strip().casefold().split())
    if not _APP_ALIAS_PATTERN.fullmatch(normalized):
        return ""
    return normalized


def validate_application(alias: object, policy: ActionPolicy) -> tuple[str, Path]:
    """Resolve an application alias to a trusted absolute executable path."""

    normalized = _normalize_app_alias(alias)
    if not normalized or normalized not in policy.allowed_applications:
        raise ActionValidationError("application_not_allowed", "That application is not in the local allowlist.")

    executable = Path(policy.allowed_applications[normalized].executable).expanduser()
    if not executable.is_absolute():
        raise ActionValidationError("invalid_application", "Configured application path must be absolute.")
    try:
        executable = executable.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ActionValidationError("application_unavailable", "The configured application is not installed.") from exc

    if not executable.is_file():
        raise ActionValidationError("application_unavailable", "The configured application is not a file.")
    if executable.name.casefold() in PROHIBITED_EXECUTABLES:
        raise ActionValidationError("application_not_allowed", "Command interpreters cannot be launched by GhostOS.")
    if executable.suffix.casefold() in PROHIBITED_EXECUTABLE_SUFFIXES:
        raise ActionValidationError("application_not_allowed", "Scripts cannot be launched as applications.")
    return normalized, executable
