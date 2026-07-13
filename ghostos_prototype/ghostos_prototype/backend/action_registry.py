"""Allowlisted local actions exposed by GhostOS.

This module contains metadata only.  It intentionally does not execute
anything: the router may select one of these names, while ``action_agent``
performs argument validation and dispatches to a fixed implementation.
Keeping the registry small and explicit prevents model output from becoming
an arbitrary command-execution interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    """Public contract for one action accepted by the local action layer."""

    name: str
    description: str
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...] = ()
    target_argument: str | None = None
    changes_filesystem: bool = False

    @property
    def allowed_arguments(self) -> frozenset[str]:
        return frozenset(self.required_arguments + self.optional_arguments)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["required_arguments"] = list(self.required_arguments)
        data["optional_arguments"] = list(self.optional_arguments)
        return data


_ACTIONS = {
    "open_file": ActionDefinition(
        name="open_file",
        description="Open an existing, permitted local file with its default app.",
        required_arguments=("path",),
        target_argument="path",
    ),
    "open_folder": ActionDefinition(
        name="open_folder",
        description="Open an existing, permitted folder in File Explorer.",
        required_arguments=("path",),
        target_argument="path",
    ),
    "open_url": ActionDefinition(
        name="open_url",
        description="Open a validated HTTP or HTTPS URL in the default browser.",
        required_arguments=("url",),
        target_argument="url",
    ),
    "open_app": ActionDefinition(
        name="open_app",
        description="Start an installed application selected from the configured allowlist.",
        required_arguments=("app",),
        target_argument="app",
    ),
    "reveal_in_explorer": ActionDefinition(
        name="reveal_in_explorer",
        description="Reveal an existing, permitted file in Windows File Explorer.",
        required_arguments=("path",),
        target_argument="path",
    ),
    "create_text_note": ActionDefinition(
        name="create_text_note",
        description="Create a new UTF-8 .txt or .md note without overwriting a file.",
        required_arguments=("path", "content"),
        target_argument="path",
        changes_filesystem=True,
    ),
    "create_folder": ActionDefinition(
        name="create_folder",
        description="Create one new folder inside an approved creation root.",
        required_arguments=("path",),
        target_argument="path",
        changes_filesystem=True,
    ),
}

# Expose a read-only view so another module cannot extend the tool surface at
# runtime based on untrusted model/user input.
ACTION_REGISTRY: Mapping[str, ActionDefinition] = MappingProxyType(_ACTIONS)


def normalize_action_name(name: object) -> str:
    """Normalize harmless spelling separators, not action semantics."""

    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def get_action(name: object) -> ActionDefinition | None:
    """Return an allowlisted definition or ``None`` for an unknown action."""

    return ACTION_REGISTRY.get(normalize_action_name(name))


def list_actions() -> list[dict]:
    """Return JSON-serializable tool metadata for a router/settings screen."""

    return [definition.to_dict() for definition in ACTION_REGISTRY.values()]

