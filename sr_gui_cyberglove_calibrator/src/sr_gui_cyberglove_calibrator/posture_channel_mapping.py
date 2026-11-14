#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2026 Gereon Buescher
# Copyright 2026 Robomantic
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation version 2 of the License.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/old-licenses/gpl-2.0.html>.
#
# SPDX-License-Identifier: GPL-2.0-only

"""Channel-to-sensor canonicalisation support for RBM posture Delta-1.x.

This module has no ROS/Qt dependency. It defines the legacy canonical order
expected by the present 16-column glove_to_human.map, validates per-device
channel assignments, performs deterministic permutation/reordering, and writes
mapping files transactionally with timestamped backups and a JSONL audit log.

The architecture intentionally separates:
    physical ADC channel -> physical sensor identity -> legacy canonical slot
so hardware solder order can vary without changing the downstream mapping.

"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml


MAPPING_VERSION = 1
DEFAULT_FILENAME = "channel_to_sensor.yaml"
HISTORY_FILENAME = ".posture_channel_mapping_history.jsonl"

# Compatibility order used by the current 16-column glove_to_human.map.
# This is deliberately named LEGACY: future named kinematic mapping must not
# need a fixed vector order, but Delta-1.x preserves it to avoid touching the
# existing remapper while introducing arbitrary hardware channel ordering.
LEGACY_CANONICAL_ORDER_16: Tuple[str, ...] = (
    "G_PinkieRingAb",
    "G_ThumbIJ",
    "G_ThumbMPJ",
    "G_IndexPIJ",
    "G_IndexMPJ",
    "G_MiddlePIJ",
    "G_MiddleMPJ",
    "G_RingPIJ",
    "G_RingMPJ",
    "G_PinkieMPJ",
    "G_PinkiePIJ",
    "G_PalmArch",
    "G_ThumbRotate",
    "G_ThumbAb",
    "G_MiddleIndexAb",
    "G_RingMiddleAb",
)


class ChannelMappingError(ValueError):
    """Raised when a mapping is unsafe or inconsistent."""


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def validate_sensor_name(name: str) -> str:
    if not isinstance(name, str):
        raise ChannelMappingError("sensor names must be strings")
    name = name.strip()
    if not name:
        raise ChannelMappingError("sensor names must not be empty")
    if any(ch in name for ch in "\n\r\t"):
        raise ChannelMappingError("sensor names must be single-line tokens")
    return name


def validate_mapping(
    canonical_order: Sequence[str],
    channel_to_sensor: Sequence[str],
    require_complete: bool = True,
) -> None:
    canonical = [validate_sensor_name(x) for x in canonical_order]
    assigned = [validate_sensor_name(x) for x in channel_to_sensor]

    if not canonical:
        raise ChannelMappingError("canonical_order must not be empty")
    if len(set(canonical)) != len(canonical):
        raise ChannelMappingError("canonical_order contains duplicate sensor names")
    if len(set(assigned)) != len(assigned):
        raise ChannelMappingError("channel_to_sensor contains duplicate sensor names")

    unknown = sorted(set(assigned) - set(canonical))
    if unknown:
        raise ChannelMappingError(
            "channel_to_sensor contains names not present in canonical_order: {}".format(
                ", ".join(unknown)
            )
        )

    if require_complete:
        if len(assigned) != len(canonical):
            raise ChannelMappingError(
                "legacy canonicalisation requires {} mapped channels, got {}".format(
                    len(canonical), len(assigned)
                )
            )
        missing = sorted(set(canonical) - set(assigned))
        if missing:
            raise ChannelMappingError(
                "channel_to_sensor is incomplete; missing: {}".format(", ".join(missing))
            )


def identity_mapping(canonical_order: Sequence[str] = LEGACY_CANONICAL_ORDER_16) -> List[str]:
    canonical = [validate_sensor_name(x) for x in canonical_order]
    validate_mapping(canonical, canonical, require_complete=True)
    return list(canonical)


def canonical_source_indices(
    canonical_order: Sequence[str], channel_to_sensor: Sequence[str]
) -> List[int]:
    """Return source channel index for each canonical output slot."""
    validate_mapping(canonical_order, channel_to_sensor, require_complete=True)
    index_by_sensor = {sensor: idx for idx, sensor in enumerate(channel_to_sensor)}
    return [index_by_sensor[sensor] for sensor in canonical_order]


def reorder_vector(
    values: Sequence[float],
    canonical_order: Sequence[str],
    channel_to_sensor: Sequence[str],
) -> List[float]:
    """Reorder one channel vector into canonical sensor order."""
    validate_mapping(canonical_order, channel_to_sensor, require_complete=True)
    if len(values) != len(channel_to_sensor):
        raise ChannelMappingError(
            "incoming vector length {} does not match mapping length {}".format(
                len(values), len(channel_to_sensor)
            )
        )
    indices = canonical_source_indices(canonical_order, channel_to_sensor)
    return [values[i] for i in indices]


def swap_assignment(channel_to_sensor: Sequence[str], channel: int, new_sensor: str) -> List[str]:
    """Safely assign *new_sensor* to *channel* by swapping with its current channel.

    This preserves a complete one-to-one permutation, avoiding transient duplicate
    names and the dangerous state where a legacy matrix slot has no source.
    """
    mapping = list(channel_to_sensor)
    if channel < 0 or channel >= len(mapping):
        raise ChannelMappingError("channel index out of range")
    new_sensor = validate_sensor_name(new_sensor)
    if new_sensor not in mapping:
        raise ChannelMappingError(
            "cannot swap to {!r}: sensor is not present in the current complete mapping".format(
                new_sensor
            )
        )
    old_sensor = mapping[channel]
    if old_sensor == new_sensor:
        return mapping
    other_channel = mapping.index(new_sensor)
    mapping[channel], mapping[other_channel] = mapping[other_channel], mapping[channel]
    return mapping


def mapping_document(
    canonical_order: Sequence[str],
    channel_to_sensor: Sequence[str],
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    validate_mapping(canonical_order, channel_to_sensor, require_complete=True)
    doc: Dict[str, object] = {
        "version": MAPPING_VERSION,
        "channel_index_base": 0,
        "canonical_order": list(canonical_order),
        "channel_to_sensor": [
            {"channel": idx, "sensor": sensor}
            for idx, sensor in enumerate(channel_to_sensor)
        ],
    }
    if metadata:
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool, list, dict)):
                clean[str(key)] = copy.deepcopy(value)
            else:
                clean[str(key)] = str(value)
        doc["metadata"] = clean
    return doc


def mapping_from_document(doc: Mapping[str, object]) -> Tuple[List[str], List[str], Dict[str, object]]:
    if not isinstance(doc, Mapping):
        raise ChannelMappingError("mapping document must be a dictionary")
    version = int(doc.get("version", 0))
    if version != MAPPING_VERSION:
        raise ChannelMappingError(
            "unsupported channel mapping version {} (expected {})".format(version, MAPPING_VERSION)
        )
    canonical = list(doc.get("canonical_order") or [])
    raw_entries = doc.get("channel_to_sensor") or []
    if not isinstance(raw_entries, list):
        raise ChannelMappingError("channel_to_sensor must be a list")

    pairs = []
    for entry in raw_entries:
        if isinstance(entry, Mapping):
            try:
                channel = int(entry["channel"])
                sensor = validate_sensor_name(entry["sensor"])
            except Exception as exc:
                raise ChannelMappingError("invalid channel_to_sensor entry: {}".format(exc))
            pairs.append((channel, sensor))
        elif isinstance(entry, str):
            # Tolerate a compact list for hand-authored files.
            pairs.append((len(pairs), validate_sensor_name(entry)))
        else:
            raise ChannelMappingError("invalid channel_to_sensor entry type")

    if sorted(ch for ch, _ in pairs) != list(range(len(pairs))):
        raise ChannelMappingError("channel indices must be contiguous and start at 0")
    pairs.sort(key=lambda x: x[0])
    mapping = [sensor for _, sensor in pairs]
    validate_mapping(canonical, mapping, require_complete=True)
    metadata = dict(doc.get("metadata") or {})
    return canonical, mapping, metadata


def load_mapping_file(
    path: str,
    fallback_canonical: Sequence[str] = LEGACY_CANONICAL_ORDER_16,
) -> Tuple[List[str], List[str], Dict[str, object], bool]:
    """Load mapping YAML; if absent return safe identity compatibility mapping.

    Returns ``canonical_order, channel_to_sensor, metadata, existed``.
    """
    if not path or not os.path.exists(path):
        canonical = list(fallback_canonical)
        return canonical, identity_mapping(canonical), {}, False
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle) or {}
    canonical, mapping, metadata = mapping_from_document(doc)
    return canonical, mapping, metadata, True


def _atomic_write_text(path: str, text: str) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".channel-map-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def safe_write_mapping(
    path: str,
    canonical_order: Sequence[str],
    channel_to_sensor: Sequence[str],
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, Optional[str]]:
    """Transactionally save a mapping with backup, re-parse and audit log."""
    if not path:
        raise ChannelMappingError("mapping target path is empty")
    validate_mapping(canonical_order, channel_to_sensor, require_complete=True)
    target = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)

    doc = mapping_document(canonical_order, channel_to_sensor, metadata=metadata)
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, allow_unicode=True)

    # Re-parse before touching the active file.
    parsed = yaml.safe_load(text)
    canonical_check, mapping_check, _ = mapping_from_document(parsed)
    if canonical_check != list(canonical_order) or mapping_check != list(channel_to_sensor):
        raise ChannelMappingError("internal mapping serialisation verification failed")

    backup = None
    if os.path.exists(target):
        backup = "{}.bak-{}".format(target, _timestamp())
        with open(target, "rb") as src, open(backup, "wb") as dst:
            dst.write(src.read())
            dst.flush()
            os.fsync(dst.fileno())

    _atomic_write_text(target, text)

    # Re-read active file after atomic replacement.
    canonical_check, mapping_check, _, _ = load_mapping_file(target, fallback_canonical=canonical_order)
    if canonical_check != list(canonical_order) or mapping_check != list(channel_to_sensor):
        raise ChannelMappingError("saved mapping did not verify after write")

    history = os.path.join(directory, HISTORY_FILENAME)
    event = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "action": "save_channel_mapping",
        "target": target,
        "backup": backup,
        "canonical_order": list(canonical_order),
        "channel_to_sensor": list(channel_to_sensor),
        "metadata": dict(metadata or {}),
    }
    with open(history, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    return {"target": target, "backup": backup, "history": history}


def infer_complete_mapping_from_names(
    incoming_names: Sequence[str],
    canonical_order: Sequence[str] = LEGACY_CANONICAL_ORDER_16,
) -> Optional[List[str]]:
    """Use incoming names only when they form the exact canonical sensor set."""
    names = [str(x) for x in incoming_names]
    canonical = list(canonical_order)
    if len(names) != len(canonical):
        return None
    if len(set(names)) != len(names):
        return None
    if set(names) != set(canonical):
        return None
    return names
