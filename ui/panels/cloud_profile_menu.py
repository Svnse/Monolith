"""Builds the ☰ cloud-profile QMenu. Pure view glue over core.cloud_profiles —
no persistence, no model loading; the panel supplies callbacks that do that."""
from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QMenu

from core import cloud_profiles as cp


def build_profile_menu(
    config: dict,
    on_switch: Callable[[str], None],
    on_save: Callable[[], None],
    on_delete: Callable[[str], None],
    parent=None,
) -> QMenu:
    menu = QMenu(parent)
    active = cp.active_id(config)
    profiles = cp.list_profiles(config)

    if not profiles:
        empty = menu.addAction("No saved profiles")
        empty.setEnabled(False)
    for prof in profiles:
        pid = prof["id"]
        model = prof.get("last_model") or prof.get("api_model") or ""
        text = f"{prof['label']}    {model}".rstrip()
        act = menu.addAction(text)
        act.setCheckable(True)
        act.setChecked(pid == active)
        act.triggered.connect(lambda _checked=False, p=pid: on_switch(p))

    menu.addSeparator()
    save_act = menu.addAction("Save current connection as profile…")
    save_act.triggered.connect(lambda _checked=False: on_save())

    if profiles:
        del_menu = menu.addMenu("Delete profile")
        for prof in profiles:
            pid = prof["id"]
            d = del_menu.addAction(prof["label"])
            d.triggered.connect(lambda _checked=False, p=pid: on_delete(p))

    return menu
