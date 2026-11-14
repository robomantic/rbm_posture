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

"""Core logic for the RBM posture calibration Delta-1 prototype.

This module deliberately has no ROS or Qt dependency so the calibration logic,
file handling, and safety checks can be tested independently from the GUI.

"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

ADC_MIN = 0.0
ADC_MAX = 4095.0

CalibrationPoint = Tuple[float, float]  # (raw ADC, calibrated angle in degree)
CalibrationMap = "OrderedDict[str, List[CalibrationPoint]]"


class CalibrationError(RuntimeError):
    """Raised when a calibration cannot be safely used or written."""


def _finite(value: float) -> bool:
    return math.isfinite(float(value))




def is_calibration_backup_path(path: str) -> bool:
    """Return True for timestamped or conventional calibration backup names."""
    name = os.path.basename(str(path or "")).lower()
    return (".cal.bak" in name) or name.endswith(".bak") or (".bak-" in name)


def recovery_path_for_backup(path: str) -> str:
    """Return a normal .cal path next to a backup without modifying the backup.

    Examples:
      bend.cal.bak-20261113-120000 -> bend.recovered.cal
      bend.cal.bak                -> bend.recovered.cal
      bend.bak                    -> bend.recovered.cal
    """
    src = os.path.abspath(os.path.expanduser(str(path)))
    directory = os.path.dirname(src)
    name = os.path.basename(src)
    lower = name.lower()
    marker = lower.find(".cal.bak")
    if marker >= 0:
        root = name[:marker]
        return os.path.join(directory, root + ".recovered.cal")
    if lower.endswith(".bak"):
        root = name[:-4]
        if root.lower().endswith(".cal"):
            root = root[:-4]
        return os.path.join(directory, root + ".recovered.cal")
    marker = lower.find(".bak-")
    if marker >= 0:
        root = name[:marker]
        if root.lower().endswith(".cal"):
            root = root[:-4]
        return os.path.join(directory, root + ".recovered.cal")
    root, ext = os.path.splitext(name)
    if ext.lower() == ".cal":
        root = root
    else:
        root = name
    return os.path.join(directory, root + ".recovered.cal")

def clone_calibration(calibration: Mapping[str, Sequence[Sequence[float]]]) -> OrderedDict:
    """Return a normalized deep copy using plain floats and insertion order."""
    out = OrderedDict()
    for name, points in calibration.items():
        out[str(name)] = [(float(p[0]), float(p[1])) for p in points]
    return out


def points_sorted_by_angle(points: Sequence[CalibrationPoint]) -> List[CalibrationPoint]:
    """Sort points by calibrated angle while preserving raw/angle pairing.

    Delta-1 intentionally supports exactly two editable points. Sorting by angle
    makes the GUI's left/right angle fields mean min/max angle regardless of raw
    sensor polarity.
    """
    return sorted([(float(r), float(a)) for r, a in points], key=lambda p: p[1])


def linear_map(raw: float, p0: CalibrationPoint, p1: CalibrationPoint) -> float:
    """Linear interpolation/extrapolation between two calibration points."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    if abs(x1 - x0) < 1e-12:
        raise CalibrationError("raw calibration points are identical")
    return y0 + (float(raw) - x0) * ((y1 - y0) / (x1 - x0))


@dataclass
class AutoRangeConfig:
    """Runtime behaviour for free-motion range adaptation."""

    persistence_s: float = 0.150
    expansion_epsilon_adc: float = 0.5
    contraction_enabled: bool = False
    contraction_rate_adc_s: float = 1.0
    # Numerical guard only. No artificial 8-count calibration margin.
    min_span_adc: float = 1e-6

    def validate(self) -> None:
        if self.persistence_s < 0.0:
            raise CalibrationError("persistence_s must be >= 0")
        if self.expansion_epsilon_adc < 0.0:
            raise CalibrationError("expansion_epsilon_adc must be >= 0")
        if self.contraction_rate_adc_s < 0.0:
            raise CalibrationError("contraction_rate_adc_s must be >= 0")
        if self.min_span_adc <= 0.0:
            raise CalibrationError("min_span_adc must be > 0")


@dataclass
class _BoundaryCandidate:
    since: Optional[float] = None
    extreme: Optional[float] = None

    def clear(self) -> None:
        self.since = None
        self.extreme = None


@dataclass
class SensorCalibrationState:
    """Two-point calibration state for one live sensor.

    point_a is always the lower calibrated angle and point_b the higher angle.
    Raw values can be in either order; this preserves sensor polarity.
    """

    name: str
    point_a: CalibrationPoint
    point_b: CalibrationPoint
    enabled: bool = True
    auto_enabled: bool = True
    supports_delta1_editing: bool = True
    raw_current: Optional[float] = None
    angle_current: Optional[float] = None
    last_auto_update_time: Optional[float] = None
    low_candidate: _BoundaryCandidate = field(default_factory=_BoundaryCandidate)
    high_candidate: _BoundaryCandidate = field(default_factory=_BoundaryCandidate)

    @classmethod
    def from_points(
        cls,
        name: str,
        points: Sequence[CalibrationPoint],
        enabled: bool = True,
        auto_enabled: bool = True,
    ) -> "SensorCalibrationState":
        if len(points) < 2:
            # Missing calibration is represented but starts disabled. The default
            # mapping is intentionally neutral and never reaches ROS until enabled.
            sorted_points = [(ADC_MIN, 0.0), (ADC_MAX, 90.0)]
            return cls(
                name=name,
                point_a=sorted_points[0],
                point_b=sorted_points[1],
                enabled=False,
                auto_enabled=False,
                supports_delta1_editing=True,
            )
        sorted_points = points_sorted_by_angle(points)
        return cls(
            name=name,
            point_a=sorted_points[0],
            point_b=sorted_points[-1],
            enabled=enabled,
            auto_enabled=auto_enabled,
            supports_delta1_editing=(len(points) <= 2),
        )

    @property
    def raw_a(self) -> float:
        return self.point_a[0]

    @property
    def angle_a(self) -> float:
        return self.point_a[1]

    @property
    def raw_b(self) -> float:
        return self.point_b[0]

    @property
    def angle_b(self) -> float:
        return self.point_b[1]

    def set_points(self, raw_a: float, angle_a: float, raw_b: float, angle_b: float) -> None:
        values = (raw_a, angle_a, raw_b, angle_b)
        if not all(_finite(v) for v in values):
            raise CalibrationError("calibration values must be finite")
        if not (ADC_MIN <= raw_a <= ADC_MAX and ADC_MIN <= raw_b <= ADC_MAX):
            raise CalibrationError("raw calibration values must be inside the 12-bit ADC range")
        if abs(raw_b - raw_a) < 1e-9:
            raise CalibrationError("raw calibration points must be different")
        pts = points_sorted_by_angle([(float(raw_a), float(angle_a)), (float(raw_b), float(angle_b))])
        self.point_a, self.point_b = pts[0], pts[1]
        self.low_candidate.clear()
        self.high_candidate.clear()

    def reset_auto_trackers(self) -> None:
        """Clear transient Auto-range candidates without changing calibration anchors."""
        self.low_candidate.clear()
        self.high_candidate.clear()
        self.last_auto_update_time = None

    def set_raw_a(self, value: float) -> None:
        self.set_points(value, self.angle_a, self.raw_b, self.angle_b)

    def set_raw_b(self, value: float) -> None:
        self.set_points(self.raw_a, self.angle_a, value, self.angle_b)

    def set_angle_a(self, value: float) -> None:
        self.set_points(self.raw_a, value, self.raw_b, self.angle_b)

    def set_angle_b(self, value: float) -> None:
        self.set_points(self.raw_a, self.angle_a, self.raw_b, value)

    def to_points(self) -> List[CalibrationPoint]:
        return [(float(self.raw_a), float(self.angle_a)), (float(self.raw_b), float(self.angle_b))]

    def local_angle(self, raw: Optional[float] = None) -> Optional[float]:
        raw = self.raw_current if raw is None else raw
        if raw is None:
            return None
        return linear_map(float(raw), self.point_a, self.point_b)

    def _raw_low_high(self) -> Tuple[int, float, int, float]:
        if self.raw_a <= self.raw_b:
            return 0, self.raw_a, 1, self.raw_b
        return 1, self.raw_b, 0, self.raw_a

    def _set_raw_by_index(self, index: int, value: float) -> None:
        value = max(ADC_MIN, min(ADC_MAX, float(value)))
        if index == 0:
            self.point_a = (value, self.angle_a)
        else:
            self.point_b = (value, self.angle_b)

    def _expand_candidate(
        self,
        candidate: _BoundaryCandidate,
        now_s: float,
        raw: float,
        is_low: bool,
        persistence_s: float,
    ) -> Optional[float]:
        if candidate.since is None:
            candidate.since = now_s
            candidate.extreme = raw
            return None
        if is_low:
            candidate.extreme = min(float(candidate.extreme), raw)
        else:
            candidate.extreme = max(float(candidate.extreme), raw)
        if (now_s - candidate.since) >= persistence_s:
            value = float(candidate.extreme)
            candidate.clear()
            return value
        return None

    def update_auto(self, raw: float, now_s: float, config: AutoRangeConfig) -> bool:
        """Update raw anchors from a live sample. Returns True on calibration change."""
        config.validate()
        self.raw_current = float(raw)
        if not (self.enabled and self.auto_enabled):
            self.low_candidate.clear()
            self.high_candidate.clear()
            self.last_auto_update_time = now_s
            return False

        changed = False
        low_idx, low, high_idx, high = self._raw_low_high()
        eps = config.expansion_epsilon_adc

        if raw < low - eps:
            value = self._expand_candidate(
                self.low_candidate, now_s, float(raw), True, config.persistence_s
            )
            if value is not None:
                max_low = high - config.min_span_adc
                value = min(value, max_low)
                self._set_raw_by_index(low_idx, value)
                changed = True
        else:
            self.low_candidate.clear()

        if raw > high + eps:
            value = self._expand_candidate(
                self.high_candidate, now_s, float(raw), False, config.persistence_s
            )
            if value is not None:
                min_high = low + config.min_span_adc
                value = max(value, min_high)
                self._set_raw_by_index(high_idx, value)
                changed = True
        else:
            self.high_candidate.clear()

        # Optional slow inward drift for prototype evaluation. Expansion is
        # persistence-gated; contraction is deliberately slow and bounded.
        if config.contraction_enabled and self.last_auto_update_time is not None:
            dt = max(0.0, min(1.0, now_s - self.last_auto_update_time))
            if dt > 0.0 and config.contraction_rate_adc_s > 0.0:
                low_idx, low, high_idx, high = self._raw_low_high()
                rate = config.contraction_rate_adc_s * dt
                # Keep only the mathematical non-zero-span guard. This is
                # deliberately much smaller than one ADC count so sensors that
                # operate at an ADC rail can use their complete measurable range.
                half_guard = max(1e-6, config.min_span_adc * 0.5)

                # Only contract a boundary if the current signal is comfortably
                # inside it. This prevents a boundary from chasing a live extremum.
                if raw > low + half_guard and (high - low) > config.min_span_adc:
                    target = min(low + rate, float(raw) - half_guard, high - config.min_span_adc)
                    if target > low:
                        self._set_raw_by_index(low_idx, target)
                        changed = True
                low_idx, low, high_idx, high = self._raw_low_high()
                if raw < high - half_guard and (high - low) > config.min_span_adc:
                    target = max(high - rate, float(raw) + half_guard, low + config.min_span_adc)
                    if target < high:
                        self._set_raw_by_index(high_idx, target)
                        changed = True

        self.last_auto_update_time = now_s
        return changed

    def collapse_to_current(self, seed_span_adc: float = 2.0) -> bool:
        """Collapse the two raw anchors around the current sample.

        The angular anchor values and raw polarity are preserved.  The span is
        intentionally tiny but non-zero so the existing affine calibration
        remains mathematically valid while free-motion Auto can expand it again.
        Near an ADC rail the seed interval is shifted fully inside 0..4095.
        """
        if self.raw_current is None or not self.supports_delta1_editing:
            return False
        raw = max(ADC_MIN, min(ADC_MAX, float(self.raw_current)))
        span = max(1e-3, float(seed_span_adc))
        span = min(span, ADC_MAX - ADC_MIN)
        low = raw - span * 0.5
        high = raw + span * 0.5
        if low < ADC_MIN:
            high += ADC_MIN - low
            low = ADC_MIN
        if high > ADC_MAX:
            low -= high - ADC_MAX
            high = ADC_MAX
        low = max(ADC_MIN, low)
        high = min(ADC_MAX, high)
        if high - low < 1e-6:
            return False

        # Preserve electrical polarity relative to the angle ordering.
        if self.raw_a <= self.raw_b:
            self.point_a = (low, self.angle_a)
            self.point_b = (high, self.angle_b)
        else:
            self.point_a = (high, self.angle_a)
            self.point_b = (low, self.angle_b)
        self.low_candidate.clear()
        self.high_candidate.clear()
        return True

    def validate(self) -> List[str]:
        errors: List[str] = []
        for label, value in (
            ("raw_a", self.raw_a),
            ("raw_b", self.raw_b),
            ("angle_a", self.angle_a),
            ("angle_b", self.angle_b),
        ):
            if not _finite(value):
                errors.append(f"{self.name}: {label} is not finite")
        if not (ADC_MIN <= self.raw_a <= ADC_MAX):
            errors.append(f"{self.name}: raw_a outside 0..4095")
        if not (ADC_MIN <= self.raw_b <= ADC_MAX):
            errors.append(f"{self.name}: raw_b outside 0..4095")
        if abs(self.raw_b - self.raw_a) < 1e-9:
            errors.append(f"{self.name}: raw calibration span is zero")
        return errors


def calibration_from_ros_param(param: Mapping[str, Sequence[Sequence[float]]]) -> OrderedDict:
    return clone_calibration(param)


def calibration_to_ros_param(calibration: Mapping[str, Sequence[CalibrationPoint]]) -> Dict[str, List[List[float]]]:
    out: Dict[str, List[List[float]]] = {}
    for name, points in calibration.items():
        out[str(name)] = [[float(r), float(a)] for r, a in points]
    return out


def parse_calibration_xml(path: str) -> OrderedDict:
    """Parse an existing Cyberglove XML calibration file."""
    try:
        root = ET.parse(path).getroot()
    except Exception as exc:
        raise CalibrationError(f"cannot parse calibration XML '{path}': {exc}") from exc
    if root.tag != "Cyberglove_calibration":
        raise CalibrationError(f"unexpected root element '{root.tag}'")
    out = OrderedDict()
    for joint_el in root.findall("Joint"):
        name = joint_el.get("name")
        if not name:
            raise CalibrationError("Joint without name")
        points: List[CalibrationPoint] = []
        for calib_el in joint_el.findall("calib"):
            try:
                raw = float(calib_el.attrib["raw_value"])
                angle = float(calib_el.attrib["calibrated_value"])
            except Exception as exc:
                raise CalibrationError(f"invalid calibration point for {name}: {exc}") from exc
            points.append((raw, angle))
        out[name] = points
    return out


def validate_calibration_map(calibration: Mapping[str, Sequence[CalibrationPoint]]) -> List[str]:
    errors: List[str] = []
    if not calibration:
        errors.append("calibration contains no joints")
        return errors
    seen = set()
    for name, points in calibration.items():
        if name in seen:
            errors.append(f"duplicate joint name: {name}")
        seen.add(name)
        if len(points) < 2:
            errors.append(f"{name}: fewer than two calibration points")
            continue
        for index, point in enumerate(points):
            if len(point) != 2:
                errors.append(f"{name}: point {index} does not contain raw and angle")
                continue
            raw, angle = float(point[0]), float(point[1])
            if not (_finite(raw) and _finite(angle)):
                errors.append(f"{name}: non-finite point {index}")
            if raw < ADC_MIN or raw > ADC_MAX:
                errors.append(f"{name}: raw point {index} outside 0..4095")
        if len(points) == 2 and abs(float(points[1][0]) - float(points[0][0])) < 1e-9:
            errors.append(f"{name}: zero raw span")
    return errors


def calibration_xml_text(
    calibration: Mapping[str, Sequence[CalibrationPoint]],
    order: Optional[Iterable[str]] = None,
) -> str:
    errors = validate_calibration_map(calibration)
    if errors:
        raise CalibrationError("invalid calibration: " + "; ".join(errors))

    if order is None:
        names = list(calibration.keys())
    else:
        names = [n for n in order if n in calibration]
        names.extend(n for n in calibration.keys() if n not in names)

    lines = ['<?xml version="1.0" ?>', '<Cyberglove_calibration>']
    for name in names:
        lines.append(f'<Joint name="{name}">')
        for raw, angle in calibration[name]:
            lines.append(
                '<calib raw_value="{:.3f}" calibrated_value="{:.3f}"/>'.format(
                    float(raw), float(angle)
                )
            )
        lines.append('</Joint>')
    lines.append('</Cyberglove_calibration>')
    return "\n".join(lines) + "\n"


def diff_calibrations(
    before: Mapping[str, Sequence[CalibrationPoint]],
    after: Mapping[str, Sequence[CalibrationPoint]],
) -> Dict[str, object]:
    changed = OrderedDict()
    all_names = list(OrderedDict.fromkeys(list(before.keys()) + list(after.keys())))
    for name in all_names:
        b = [(float(x), float(y)) for x, y in before.get(name, [])]
        a = [(float(x), float(y)) for x, y in after.get(name, [])]
        if b != a:
            changed[name] = {"before": b, "after": a}
    return {"changed_count": len(changed), "joints": changed}


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_write_calibration(
    target_path: str,
    calibration: Mapping[str, Sequence[CalibrationPoint]],
    order: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Dict[str, Optional[str]]:
    """Safely replace a calibration file with backup, validation and history.

    The temporary file is written and parsed in the same directory, then moved
    into place with os.replace(). The previous active file is preserved as a
    timestamped .bak file. A JSONL history record is appended after success.
    """
    target_path = os.path.abspath(os.path.expanduser(target_path))
    directory = os.path.dirname(target_path) or os.curdir
    if not os.path.isdir(directory):
        raise CalibrationError(f"target directory does not exist: {directory}")

    text = calibration_xml_text(calibration, order=order)
    stamp = _timestamp()
    backup_path: Optional[str] = None
    previous = OrderedDict()
    if os.path.exists(target_path):
        try:
            previous = parse_calibration_xml(target_path)
        except CalibrationError:
            # Even a malformed old file is valuable as a byte-for-byte backup.
            previous = OrderedDict()
        backup_path = f"{target_path}.bak-{stamp}"
        shutil.copy2(target_path, backup_path)

    fd, temp_path = tempfile.mkstemp(
        prefix="." + os.path.basename(target_path) + ".tmp-", dir=directory, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # Parse the exact bytes that will become active.
        parsed = parse_calibration_xml(temp_path)
        errors = validate_calibration_map(parsed)
        if errors:
            raise CalibrationError("temporary file validation failed: " + "; ".join(errors))
        os.replace(temp_path, target_path)
        temp_path = ""
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    history_path = os.path.join(directory, ".posture_calibration_history.jsonl")
    record = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "action": "save_calibration",
        "target": target_path,
        "backup": backup_path,
        "diff": diff_calibrations(previous, calibration),
        "metadata": dict(metadata or {}),
    }
    try:
        with open(history_path, "a", encoding="utf-8") as history:
            history.write(json.dumps(record, sort_keys=True) + "\n")
            history.flush()
            os.fsync(history.fileno())
    except OSError:
        # Saving the calibration succeeded. Failure to append the optional audit
        # log must not invalidate the atomic update.
        history_path = None

    return {
        "target": target_path,
        "backup": backup_path,
        "history": history_path,
    }


def merge_live_states(
    base: Mapping[str, Sequence[CalibrationPoint]],
    states: Mapping[str, SensorCalibrationState],
) -> OrderedDict:
    """Merge enabled two-point live states into a complete calibration map.

    Disabled sensors and source calibrations with >2 points are preserved from
    base. This is critical for Delta-1: the tool must not silently delete legacy
    or non-live calibration entries.
    """
    out = clone_calibration(base)
    for name, state in states.items():
        if not state.enabled:
            continue
        if not state.supports_delta1_editing and name in out and len(out[name]) > 2:
            continue
        out[name] = state.to_points()
    return out
