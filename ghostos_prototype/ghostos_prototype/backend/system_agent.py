"""
agents/system_agent.py
System Agent - this is the one box from the v2.0 architecture / agent
router diagram that had ZERO real implementation before this pass (the
audit scored both "System Agent" and "System Actions" at 0%). This file
only builds the *monitoring* half - "how much RAM/disk do I have" - not
the *actions* half (open app, shutdown, toggle Bluetooth, launch VS
Code). Actions are a meaningfully different and larger piece of work
(OS-level automation, permissions, safety review for anything that can
shut down or modify the user's machine) and are deliberately still out of
scope here - this is monitoring only, and the system prompt still tells
the model it can't take actions.

Uses psutil for CPU/RAM/battery if it's installed (added to
requirements.txt); disk usage works via the standard library either way,
so this degrades gracefully on a machine that hasn't pip-installed psutil
yet instead of hard-failing.
"""

import shutil
from pathlib import Path

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

SYSTEM_INTENT_KEYWORDS = {
    "cpu", "ram", "disk space", "storage left", "storage space", "battery",
    "system stats", "system performance", "how much space", "free space",
    "memory usage", "ram usage", "cpu usage", "disk usage", "how full",
}


def wants_system_info(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SYSTEM_INTENT_KEYWORDS)


def get_system_stats() -> dict:
    """Best-effort system snapshot. Disk usage always works (stdlib);
    CPU/RAM/battery only populate if psutil is installed."""
    stats = {"psutil_available": _HAS_PSUTIL}

    try:
        total, used, free = shutil.disk_usage(Path.home().anchor or "/")
        stats["disk"] = {
            "total_gb": round(total / (1024 ** 3), 1),
            "used_gb": round(used / (1024 ** 3), 1),
            "free_gb": round(free / (1024 ** 3), 1),
            "percent_used": round(used / total * 100, 1) if total else None,
        }
    except Exception as e:
        stats["disk"] = None
        stats["disk_error"] = str(e)

    if _HAS_PSUTIL:
        try:
            stats["cpu_percent"] = psutil.cpu_percent(interval=0.3)
            vm = psutil.virtual_memory()
            stats["ram"] = {
                "total_gb": round(vm.total / (1024 ** 3), 1),
                "used_gb": round(vm.used / (1024 ** 3), 1),
                "percent_used": vm.percent,
            }
            battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
            if battery:
                stats["battery_percent"] = battery.percent
        except Exception as e:
            stats["psutil_error"] = str(e)

    return stats


def system_agent(query: str) -> str | None:
    """Returns a '[Live system stats]' context block for the router's
    system_query intent, or None if there's nothing to report. Safe to
    call defensively even outside that intent - it no-ops on non-system
    queries via wants_system_info()."""
    if not wants_system_info(query):
        return None

    stats = get_system_stats()
    lines = []

    disk = stats.get("disk")
    if disk:
        lines.append(
            f"Disk: {disk['used_gb']} GB used of {disk['total_gb']} GB "
            f"({disk['percent_used']}% full, {disk['free_gb']} GB free)"
        )

    if stats.get("psutil_available"):
        if "cpu_percent" in stats:
            lines.append(f"CPU usage: {stats['cpu_percent']}%")
        if "ram" in stats:
            ram = stats["ram"]
            lines.append(f"RAM: {ram['used_gb']} GB used of {ram['total_gb']} GB ({ram['percent_used']}%)")
        if "battery_percent" in stats:
            lines.append(f"Battery: {stats['battery_percent']}%")
    else:
        lines.append(
            "(CPU/RAM/battery unavailable - psutil isn't installed on this "
            "machine; disk usage above is always available via the "
            "standard library regardless)"
        )

    if not lines:
        return None
    return "[Live system stats - current readings, not a file]\n" + "\n".join(lines)
