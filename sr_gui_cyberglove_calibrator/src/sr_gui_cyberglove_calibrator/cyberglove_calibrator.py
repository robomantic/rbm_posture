#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright 2011 Shadow Robot Company Ltd.
# Copyright 2026 Gereon Buescher
# Copyright 2026 Robomantic
#
# This file was substantially modified in 2026 by Gereon Buescher and Robomantic.
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

"""RBM posture calibration GUI — Delta-1 prototype.

Drop-in replacement for the historical gesture-based Cyberglove calibrator.
The rqt plugin class name is intentionally unchanged so existing RViz launch
code continues to work without modification.

Design goals for Delta-1:
  * discover live bend sensors from /<ns>/raw/joint_states (no 16/18/22 list),
  * show raw 12-bit ADC and calibrated angle together,
  * edit two raw<->angle mapping anchors directly,
  * optionally expand / slowly contract raw anchors during free hand motion,
  * push work-in-progress calibration through the existing ROS param +
    reload_calibration service so RViz provides live feedback,
  * never overwrite the active .cal file without timestamped backup,
    XML re-parse validation, atomic replacement, and an audit log,
  * preserve disabled, non-live, or legacy multi-point calibrations unchanged.

No new third-party GUI/runtime library is introduced. The module uses only the
ROS Noetic / rqt / python_qt_binding stack already used by this package and the
Python standard library.

"""

from __future__ import annotations

import copy
import math
import os
import re
import threading
import sys
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import rospy
import rospkg
import rosservice
from qt_gui.plugin import Plugin
from python_qt_binding import QtCore, QtGui, QtWidgets
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from sr_gui_cyberglove_calibrator.posture_channel_mapping import (
    LEGACY_CANONICAL_ORDER_16,
    ChannelMappingError,
    load_mapping_file,
    safe_write_mapping,
    swap_assignment,
    validate_mapping,
)

from sr_gui_cyberglove_calibrator.posture_hand_svg import HandMapDialog, HandSvgWidget, resource_path

from sr_gui_cyberglove_calibrator.posture_calibration_core import (
    ADC_MAX,
    ADC_MIN,
    AutoRangeConfig,
    CalibrationError,
    SensorCalibrationState,
    calibration_from_ros_param,
    calibration_to_ros_param,
    clone_calibration,
    merge_live_states,
    parse_calibration_xml,
    is_calibration_backup_path,
    recovery_path_for_backup,
    safe_write_calibration,
    validate_calibration_map,
)

try:
    Signal = QtCore.Signal
except AttributeError:  # PyQt5 fallback
    Signal = QtCore.pyqtSignal


# ---------------------------------------------------------------------------
# Delta-1 version / tunables
# ---------------------------------------------------------------------------
PLUGIN_VERSION = "0.1.12-delta1.12"
SOURCE_RAW_TOPIC_SUFFIX = "raw/joint_states"
RAW_TOPIC_SUFFIX = "raw/joint_states_canonical"
CALIBRATED_TOPIC_SUFFIX = "calibrated/joint_states"
CALIBRATION_PARAM_SUFFIX = "cyberglove/calibration"
RELOAD_SERVICE_SUFFIX = "/reload_calibration"
CHANNEL_CANONICALIZER_NODE = "posture_channel_canonicalizer"
CHANNEL_RELOAD_SERVICE_SUFFIX = "/reload_mapping"

UI_TICK_MS = 50
DEFAULT_APPLY_HZ = 2.0
DEFAULT_PERSISTENCE_MS = 150
DEFAULT_NARROW_RATE_ADC_S = 1.0
DEFAULT_MIN_SPAN_ADC = 1e-6

# Stable UI colours.  These deliberately do not reuse the native Qt selection
# colour: calibration flags must remain visually distinct from row selection.
FLAG_BLUE = QtGui.QColor(25, 92, 150)
ROW_BASE = QtGui.QColor(255, 255, 255)
ROW_ALT = QtGui.QColor(244, 244, 244)
ROW_HOVER = QtGui.QColor(218, 238, 250)
ROW_SELECTED = QtGui.QColor(178, 215, 240)

# Debug / operation keys. These are deliberately centralized so later versions
# can expose them in help text or remap them without touching widget code.
KEY_TOGGLE_CALIB = "Space"
KEY_PREVIOUS_SENSOR = "Up"
KEY_NEXT_SENSOR = "Down"
KEY_TOGGLE_SENSOR_AUTO = "Return"
KEY_SAVE_ACTIVE = "Ctrl+S"
KEY_RESET_SESSION = "Ctrl+R"
KEY_TOGGLE_DEBUG = "F12"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _normalize_namespace(ns: str) -> str:
    ns = ns or "/"
    if not ns.startswith("/"):
        ns = "/" + ns
    if not ns.endswith("/"):
        ns += "/"
    return ns


def _topic(ns: str, suffix: str) -> str:
    return _normalize_namespace(ns) + suffix.lstrip("/")


def _monospace_font(base: QtGui.QFont) -> QtGui.QFont:
    font = QtGui.QFont(base)
    font.setStyleHint(QtGui.QFont.Monospace)
    try:
        font.setFamily("Monospace")
    except Exception:
        pass
    return font


def _clear_layout(layout: QtWidgets.QLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if widget is not None:
            widget.deleteLater()
        elif child_layout is not None:
            _clear_layout(child_layout)


class _GuiSignals(QtCore.QObject):
    inventory_changed = Signal(object)
    metadata_changed = Signal()


class SensorIdentityComboBox(QtWidgets.QComboBox):
    """Channel identity selector with non-destructive SVG hover preview.

    Merely moving through the popup must never change the live mapping.  It only
    emits the candidate identity so the linked SVG can highlight where that
    sensor belongs.  The current assignment is emitted when the popup opens,
    allowing the selected sensor and the hovered replacement to use two colours.
    """

    sensorHovered = Signal(object)
    popupOpened = Signal(object)

    def __init__(self, parent=None):
        super(SensorIdentityComboBox, self).__init__(parent)
        view = self.view()
        view.setMouseTracking(True)
        view.entered.connect(self._on_popup_entered)
        view.viewport().installEventFilter(self)

    def showPopup(self):  # noqa: N802
        self.popupOpened.emit(self.currentText())
        super(SensorIdentityComboBox, self).showPopup()

    def hidePopup(self):  # noqa: N802
        self.sensorHovered.emit(None)
        super(SensorIdentityComboBox, self).hidePopup()

    def _on_popup_entered(self, index) -> None:
        if index is not None and index.isValid():
            self.sensorHovered.emit(self.itemText(index.row()))

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is self.view().viewport() and event.type() == QtCore.QEvent.Leave:
            self.sensorHovered.emit(None)
        return super(SensorIdentityComboBox, self).eventFilter(obj, event)


# ---------------------------------------------------------------------------
# Bar widget
# ---------------------------------------------------------------------------
class ElidedLabel(QtWidgets.QLabel):
    """QLabel that elides long text instead of forcing the window wider."""

    def __init__(self, text="", parent=None):
        super(ElidedLabel, self).__init__(text, parent)
        self._full_text = str(text)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(40)

    def setFullText(self, text: str) -> None:
        self._full_text = str(text or "")
        self.setToolTip(self._full_text)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        metrics = QtGui.QFontMetrics(self.font())
        text = metrics.elidedText(self._full_text, QtCore.Qt.ElideMiddle, max(1, self.width() - 4))
        # WindowText is used deliberately: status/header labels customise that
        # role (amber/red/blue/quiet grey) without affecting editable text.
        painter.setPen(self.palette().color(QtGui.QPalette.WindowText))
        painter.drawText(self.rect().adjusted(2, 0, -2, 0), int(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft), text)


class SelectablePathLine(QtWidgets.QLineEdit):
    """Read-only, selectable path/status field that never forces window width."""

    def __init__(self, text="", parent=None):
        super(SelectablePathLine, self).__init__(parent)
        self.setReadOnly(True)
        self.setFrame(False)
        self.setText(str(text or ""))
        self.setMinimumWidth(40)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.Text, palette.color(QtGui.QPalette.Mid))
        palette.setColor(QtGui.QPalette.Base, QtCore.Qt.transparent)
        self.setPalette(palette)
        self.setStyleSheet("QLineEdit { background: transparent; border: 0px; padding: 0px; }")

    def setFullText(self, text: str) -> None:
        value = str(text or "")
        # Do not reset the cursor/selection on every periodic GUI refresh;
        # users must be able to select and copy a long path reliably.
        if self.text() != value:
            self.setText(value)
            self.setCursorPosition(0)
        self.setToolTip(value)


class MappingBar(QtWidgets.QWidget):
    """Compact live mapping bar.

    Raw mode always spans the full 12-bit ADC domain and supports direct mouse
    dragging of the two calibration anchors. Angle mode shows the physical
    endpoint values and deliberately leaves side margins so extrapolation can be
    seen rather than clipped.
    """

    RAW = "raw"
    ANGLE = "angle"
    anchorMoved = Signal(int, float)

    def __init__(self, mode: str, parent=None):
        super(MappingBar, self).__init__(parent)
        self.mode = mode
        self.current: Optional[float] = None
        self.anchor_a: Optional[float] = None
        self.anchor_b: Optional[float] = None
        self.range_a: float = 0.0
        self.range_b: float = 90.0
        self._drag_anchor: Optional[int] = None
        self.setMinimumWidth(190)
        self.setMinimumHeight(15 if mode == self.RAW else 18)
        self.setMaximumHeight(18 if mode == self.RAW else 21)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        if mode == self.RAW:
            self.setCursor(QtCore.Qt.PointingHandCursor)
            self.setToolTip("Raw ADC 0…4095. Blue flags are the two calibration anchors; drag a flag to edit it.")
        else:
            self.setToolTip("Calibrated angle. Orange extension means the current value lies outside the calibrated angular range.")

    def set_raw(self, current: Optional[float], anchor_a: float, anchor_b: float) -> None:
        self.current = current
        self.anchor_a = float(anchor_a)
        self.anchor_b = float(anchor_b)
        self.update()

    def set_angle(self, current: Optional[float], angle_a: float, angle_b: float) -> None:
        self.current = current
        self.range_a = float(angle_a)
        self.range_b = float(angle_b)
        self.update()

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    def _raw_geometry(self):
        left = 5.0
        right = max(left + 20.0, float(self.width()) - 5.0)
        return left, right

    def _x_raw(self, value: float) -> float:
        left, right = self._raw_geometry()
        fraction = (float(value) - ADC_MIN) / (ADC_MAX - ADC_MIN)
        return left + self._clamp(fraction, 0.0, 1.0) * (right - left)

    def _raw_from_x(self, x: float) -> float:
        left, right = self._raw_geometry()
        f = self._clamp((float(x) - left) / max(1.0, right - left), 0.0, 1.0)
        return round(ADC_MIN + f * (ADC_MAX - ADC_MIN))

    def mousePressEvent(self, event):  # noqa: N802
        if self.mode != self.RAW or self.anchor_a is None or self.anchor_b is None:
            return super(MappingBar, self).mousePressEvent(event)
        x = float(event.pos().x())
        xa, xb = self._x_raw(self.anchor_a), self._x_raw(self.anchor_b)
        self._drag_anchor = 0 if abs(x-xa) <= abs(x-xb) else 1
        if min(abs(x-xa), abs(x-xb)) <= 14.0:
            # Explicit mouse grab prevents QTableWidget from stealing a drag
            # after the initial press when the pointer crosses cell boundaries.
            self.grabMouse()
            self.anchorMoved.emit(self._drag_anchor, self._raw_from_x(x))
        else:
            self._drag_anchor = None
        event.accept()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self.mode == self.RAW and self._drag_anchor is not None:
            self.anchorMoved.emit(self._drag_anchor, self._raw_from_x(float(event.pos().x())))
            event.accept()
            return
        super(MappingBar, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._drag_anchor = None
        try:
            self.releaseMouse()
        except Exception:
            pass
        event.accept()

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        palette = self.palette()
        width = float(self.width())
        height = float(self.height())
        y = height * 0.52
        pen_base = QtGui.QPen(palette.color(QtGui.QPalette.Mid), 1)
        warning = QtGui.QColor(220, 120, 0)

        if self.mode == self.RAW:
            left, right = self._raw_geometry()
            painter.setPen(pen_base)
            painter.drawLine(int(left), int(y), int(right), int(y))
            for idx, anchor in enumerate((self.anchor_a, self.anchor_b)):
                if anchor is None:
                    continue
                x = self._x_raw(anchor)
                painter.setBrush(FLAG_BLUE)
                painter.setPen(QtCore.Qt.NoPen)
                if idx == 0:
                    tri = QtGui.QPolygonF([QtCore.QPointF(x-4, y-7), QtCore.QPointF(x+4, y-7), QtCore.QPointF(x, y-1)])
                else:
                    tri = QtGui.QPolygonF([QtCore.QPointF(x-4, y+7), QtCore.QPointF(x+4, y+7), QtCore.QPointF(x, y+1)])
                painter.drawPolygon(tri)
            if self.current is not None and math.isfinite(float(self.current)):
                x = self._x_raw(self.current)
                painter.setBrush(palette.color(QtGui.QPalette.Text))
                painter.setPen(QtCore.Qt.NoPen)
                painter.drawEllipse(QtCore.QPointF(x, y), 4.0, 4.0)
            return

        lo = min(self.range_a, self.range_b)
        hi = max(self.range_a, self.range_b)
        span = max(1e-9, hi-lo)
        fm = QtGui.QFontMetrics(painter.font())
        left_text = "{:g}°".format(lo)
        right_text = "{:g}°".format(hi)
        left = float(fm.horizontalAdvance(left_text) + 7)
        right = max(left + 30.0, width - float(fm.horizontalAdvance(right_text) + 7))

        # Layer order: base line -> overshoot -> current dot -> text.  This keeps
        # orange over-range feedback from obscuring endpoint labels.
        painter.setPen(pen_base)
        painter.drawLine(int(left), int(y), int(right), int(y))
        painter.drawLine(int(left), int(y-4), int(left), int(y+4))
        painter.drawLine(int(right), int(y-4), int(right), int(y+4))
        if self.current is not None and math.isfinite(float(self.current)):
            value = float(self.current)
            x = left + ((value-lo)/span) * (right-left)
            x_clamped = self._clamp(x, 3.0, width-3.0)
            if value < lo:
                painter.setPen(QtGui.QPen(warning, 2))
                painter.drawLine(int(x_clamped), int(y), int(left), int(y))
                painter.setBrush(warning)
            elif value > hi:
                painter.setPen(QtGui.QPen(warning, 2))
                painter.drawLine(int(right), int(y), int(x_clamped), int(y))
                painter.setBrush(warning)
            else:
                painter.setBrush(palette.color(QtGui.QPalette.Text))
            painter.setPen(QtCore.Qt.NoPen)
            painter.drawEllipse(QtCore.QPointF(x_clamped, y), 4.0, 4.0)

        painter.setPen(palette.color(QtGui.QPalette.Text))
        painter.drawText(0, int(y+4), left_text)
        painter.drawText(int(width-fm.horizontalAdvance(right_text)), int(y+4), right_text)


class RowCellWidget(QtWidgets.QWidget):
    """Cell host that paints row colour without styling native child controls.

    Qt style sheets applied to a parent propagate into combo/spin/button children
    on Ubuntu themes.  Painting the host background directly keeps platform
    controls native while making hover/selection/zebra feedback unambiguous.
    """
    def __init__(self, parent=None):
        super(RowCellWidget, self).__init__(parent)
        self._row_colour = ROW_BASE
        self.setAutoFillBackground(False)

    def set_row_colour(self, colour) -> None:
        self._row_colour = QtGui.QColor(colour)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self._row_colour)
        painter.end()
        super(RowCellWidget, self).paintEvent(event)


class SensorRow(QtCore.QObject):
    """Controller for one *real QTableWidget row*.

    The physical ADC channel is the row identity.  Each column owns one compact
    widget, so Qt's table geometry (not a parallel pseudo-grid) controls alignment,
    resizing, hiding and alternating row backgrounds.  Calibration state remains
    attached to the assigned named sensor; a safe channel-map swap therefore moves
    the corresponding state to the other physical-channel row on rebuild.
    """
    changed = Signal(object)
    selected = Signal(object)
    hovered = Signal(object)
    mappingRequested = Signal(int, object)
    mappingPopupOpened = Signal(int, object)
    mappingCandidateHovered = Signal(object)
    resetRequested = Signal(object)

    CHANNEL_W = 48
    SENSOR_W = 158
    CALIBRATION_W = 205
    CURRENT_W = 80
    EXACT_W = 136
    AUTO_W = 54
    ROW_H = 49

    def __init__(self, channel: int, state: SensorCalibrationState, baseline_points,
                 canonical_order, mapping_available: bool = False,
                 remap_unlocked: bool = False, parent=None):
        super(SensorRow, self).__init__(parent)
        self.channel = int(channel)
        self.state = state
        self.baseline_points = copy.deepcopy(list(baseline_points))
        self._canonical_order = list(canonical_order)
        self._mapping_available = bool(mapping_available)
        self._remap_unlocked = bool(remap_unlocked)
        self._block_ui = False
        self._selected = False
        self._editable_two_point = state.supports_delta1_editing
        self._sensor_master = True
        self._auto_master = True
        self._svg_hovered = False
        self._table_hovered = False
        self._alternate = False
        self._source_label = "—"
        self._table = None
        self._table_row = -1

        self.use_box = QtWidgets.QCheckBox(str(self.channel))
        self.use_box.setChecked(bool(state.enabled))
        self.use_box.setToolTip(
            "Include this channel's assigned sensor in calibration/editing. "
            "The channel remains present in the legacy 16-slot processing chain."
        )
        self.use_cell = self._center_cell(self.use_box, left=True)

        self.sensor_combo = SensorIdentityComboBox()
        self.sensor_combo.addItems(self._canonical_order)
        idx = self.sensor_combo.findText(state.name)
        if idx >= 0:
            self.sensor_combo.setCurrentIndex(idx)
        self.sensor_combo.setToolTip(
            "Physical channel → sensor identity. Remapping is locked by default; "
            "unlock it in the Sensor header. Existing identities are swapped safely."
        )
        self.sensor_combo.setStyleSheet(
            "QComboBox:disabled { color: palette(text); background: palette(base); }"
        )
        self.sensor_label = QtWidgets.QLabel(state.name)
        self.sensor_label.setToolTip(self.sensor_combo.toolTip())
        self.sensor_stack = QtWidgets.QStackedWidget()
        self.sensor_stack.addWidget(self.sensor_label)
        self.sensor_stack.addWidget(self.sensor_combo)
        self.sensor_cell = RowCellWidget()
        sl = QtWidgets.QHBoxLayout(self.sensor_cell)
        sl.setContentsMargins(2, 1, 2, 1); sl.setSpacing(0)
        sl.addWidget(self.sensor_stack)

        self.bars_widget = RowCellWidget()
        bars = QtWidgets.QVBoxLayout(self.bars_widget)
        bars.setContentsMargins(2, 0, 2, 0); bars.setSpacing(0)
        self.raw_bar = MappingBar(MappingBar.RAW)
        self.angle_bar = MappingBar(MappingBar.ANGLE)
        bars.addWidget(self.raw_bar); bars.addWidget(self.angle_bar)

        self.current_widget = RowCellWidget()
        current = QtWidgets.QVBoxLayout(self.current_widget)
        current.setContentsMargins(1, 0, 1, 0); current.setSpacing(0)
        self.raw_current = QtWidgets.QLabel("—")
        self.angle_current = QtWidgets.QLabel("—")
        for label in (self.raw_current, self.angle_current):
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            label.setFont(_monospace_font(label.font()))
        self.raw_current.setToolTip("Current raw 12-bit ADC value")
        self.angle_current.setToolTip("Current calibrated angle")
        current.addWidget(self.raw_current); current.addWidget(self.angle_current)

        self.exact_widget = RowCellWidget()
        exact = QtWidgets.QGridLayout(self.exact_widget)
        exact.setContentsMargins(1, 0, 1, 0); exact.setSpacing(2)
        self.raw_a = self._raw_spin("Raw ADC mapped to the minimum angle")
        self.raw_b = self._raw_spin("Raw ADC mapped to the maximum angle")
        self.angle_a = self._angle_spin("Minimum calibrated angle")
        self.angle_b = self._angle_spin("Maximum calibrated angle")
        exact.addWidget(self.raw_a, 0, 0); exact.addWidget(self.raw_b, 0, 1)
        exact.addWidget(self.angle_a, 1, 0); exact.addWidget(self.angle_b, 1, 1)

        self.auto_box = QtWidgets.QCheckBox()
        self.auto_box.setChecked(bool(state.auto_enabled and self._editable_two_point))
        self.auto_box.setToolTip("Include this sensor in free-motion automatic range adaptation")
        self.reset_button = QtWidgets.QToolButton()
        self.reset_button.setText("↺")
        self.reset_button.setToolTip("Restore this sensor to the session reference calibration")
        self.reset_button.setFixedWidth(24)
        self.auto_widget = RowCellWidget()
        auto = QtWidgets.QHBoxLayout(self.auto_widget)
        auto.setContentsMargins(1, 0, 1, 0); auto.setSpacing(2)
        auto.addStretch(1); auto.addWidget(self.auto_box); auto.addWidget(self.reset_button); auto.addStretch(1)

        self._cell_widgets = [
            self.use_cell, self.sensor_cell, self.bars_widget,
            self.current_widget, self.exact_widget, self.auto_widget,
        ]
        # Do not use a transparent stylesheet on these containers. Qt style
        # sheets propagate to child controls and caused Ubuntu/Qt to render
        # combo boxes, spin boxes and tool buttons as black rectangles. Row
        # colours are applied through QPalette instead.
        # RowCellWidget paints zebra/hover/selected background itself; native
        # child controls keep their Ubuntu/Qt appearance.

        self._wire_signals()
        self.sync_editors_from_state()
        self._update_enabled_state()
        self.set_remap_enabled(self._remap_unlocked)
        if not self._editable_two_point:
            self.auto_box.setChecked(False)
            self.auto_box.setEnabled(False)
            self.use_box.setToolTip(
                "This calibration has more than two points and is preserved read-only in Delta-1.12."
            )

    @staticmethod
    def _center_cell(widget, left=False):
        host = RowCellWidget()
        lay = QtWidgets.QHBoxLayout(host)
        lay.setContentsMargins(2, 0, 2, 0); lay.setSpacing(0)
        if not left:
            lay.addStretch(1)
        lay.addWidget(widget)
        lay.addStretch(1)
        return host

    def install_into_table(self, table: QtWidgets.QTableWidget, row_index: int) -> None:
        self._table = table
        self._table_row = int(row_index)
        for col, widget in enumerate(self._cell_widgets):
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            table.setItem(row_index, col, item)
            table.setCellWidget(row_index, col, widget)
        table.setRowHeight(row_index, self.ROW_H)
        self._update_row_background()

    def _raw_spin(self, tooltip: str) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(ADC_MIN, ADC_MAX); box.setDecimals(0); box.setSingleStep(10.0)
        box.setKeyboardTracking(False); box.setLocale(QtCore.QLocale.c())
        box.setMinimumWidth(58); box.setMaximumWidth(68); box.setFixedHeight(21); box.setToolTip(tooltip)
        return box

    def _angle_spin(self, tooltip: str) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(-360.0, 360.0); box.setDecimals(1); box.setSingleStep(1.0)
        box.setKeyboardTracking(False); box.setLocale(QtCore.QLocale.c()); box.setSuffix("°")
        box.setMinimumWidth(58); box.setMaximumWidth(68); box.setFixedHeight(21); box.setToolTip(tooltip)
        return box

    def _wire_signals(self) -> None:
        self.use_box.toggled.connect(self._on_use_toggled)
        self.auto_box.toggled.connect(self._on_auto_toggled)
        for box in (self.raw_a, self.raw_b, self.angle_a, self.angle_b):
            box.valueChanged.connect(self._on_editor_changed)
        self.raw_bar.anchorMoved.connect(self._on_anchor_dragged)
        self.reset_button.clicked.connect(lambda: self.resetRequested.emit(self.state.name))
        self.sensor_combo.currentTextChanged.connect(lambda sensor: self.mappingRequested.emit(self.channel, sensor))
        self.sensor_combo.popupOpened.connect(lambda sensor: self.mappingPopupOpened.emit(self.channel, sensor))
        self.sensor_combo.sensorHovered.connect(self.mappingCandidateHovered.emit)
        for widget in (self.use_box, self.sensor_label, self.sensor_combo, self.sensor_cell,
                       self.raw_bar, self.angle_bar, self.bars_widget, self.current_widget,
                       self.raw_current, self.angle_current, self.exact_widget, self.raw_a,
                       self.raw_b, self.angle_a, self.angle_b, self.auto_widget, self.auto_box,
                       self.reset_button):
            widget.installEventFilter(self)

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self.selected.emit(self.state.name)
        elif event.type() == QtCore.QEvent.Enter:
            self.hovered.emit(self.state.name)
        return super(SensorRow, self).eventFilter(obj, event)

    def setFocus(self, reason=QtCore.Qt.OtherFocusReason) -> None:  # noqa: N802
        # Focus only. Scrolling is source-aware in the parent controller: SVG
        # clicks and keyboard navigation scroll deliberately, ordinary table
        # clicks/flag drags/reset presses do not jump the viewport.
        if self._table is not None:
            self._table.setFocus(reason)

    def set_source_label(self, label: str) -> None:
        self._source_label = str(label or "—")
        self.use_box.setToolTip(
            "ADC channel {} · driver label: {}\nInclude this channel's assigned sensor in calibration/editing.".format(
                self.channel, self._source_label
            )
        )
        self.sensor_combo.setToolTip(
            "Driver label: {}\nPhysical channel → sensor identity. Unlock Remap in the header to edit.".format(
                self._source_label
            )
        )

    def set_remap_enabled(self, enabled: bool) -> None:
        self._remap_unlocked = bool(enabled)
        active = bool(self._mapping_available and self._remap_unlocked and self._sensor_master and self.state.enabled)
        self.sensor_combo.setEnabled(active)
        self.sensor_label.setText(self.state.name)
        self.sensor_stack.setCurrentWidget(self.sensor_combo if active else self.sensor_label)

    def set_mapping_available(self, available: bool) -> None:
        self._mapping_available = bool(available)
        self.set_remap_enabled(self._remap_unlocked)

    def set_column_visibility(self, calibration: bool, current: bool, exact: bool) -> None:
        # Column visibility is owned by QTableWidget in Delta-1.12.  Keeping this
        # compatibility method avoids coupling the row controller to the view.
        return None

    def set_alternate(self, alternate: bool) -> None:
        self._alternate = bool(alternate)
        self._update_row_background()

    def set_svg_hovered(self, hovered: bool) -> None:
        self._svg_hovered = bool(hovered)
        self._update_row_background()

    def set_table_hovered(self, hovered: bool) -> None:
        self._table_hovered = bool(hovered)
        self._update_row_background()

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self._update_row_background()

    def _update_row_background(self) -> None:
        if self._table is None or self._table_row < 0:
            return
        if self._selected:
            colour = ROW_SELECTED
        elif self._svg_hovered or self._table_hovered:
            colour = ROW_HOVER
        elif self._alternate:
            colour = ROW_ALT
        else:
            colour = ROW_BASE

        brush = QtGui.QBrush(colour)
        for col, widget in enumerate(self._cell_widgets):
            item = self._table.item(self._table_row, col)
            if item is not None:
                item.setBackground(brush)
            # Cell widgets cover the item's paint area, therefore give only the
            # container a row-colour palette. Native child controls keep their
            # platform style and remain clearly visible.
            if hasattr(widget, "set_row_colour"):
                widget.set_row_colour(colour)
            else:
                palette = widget.palette()
                palette.setColor(QtGui.QPalette.Window, colour)
                widget.setPalette(palette)
                widget.setAutoFillBackground(True)

    def set_master_states(self, sensor_master: bool, auto_master: bool) -> None:
        self._sensor_master = bool(sensor_master); self._auto_master = bool(auto_master)
        self._update_enabled_state()

    def _on_use_toggled(self, checked: bool) -> None:
        if self._block_ui: return
        self.state.enabled = bool(checked); self._update_enabled_state(); self.changed.emit(self.state.name)

    def _on_auto_toggled(self, checked: bool) -> None:
        if self._block_ui: return
        self.state.auto_enabled = bool(checked and self._editable_two_point); self.changed.emit(self.state.name)

    def _update_enabled_state(self) -> None:
        sensor_active = bool(self._sensor_master and self.state.enabled and self._editable_two_point)
        self.raw_bar.setEnabled(sensor_active); self.angle_bar.setEnabled(sensor_active)
        self.exact_widget.setEnabled(sensor_active); self.current_widget.setEnabled(sensor_active)
        self.reset_button.setEnabled(sensor_active)
        self.auto_box.setEnabled(bool(sensor_active and self._auto_master))
        self.use_box.setEnabled(bool(self._sensor_master))
        self.set_remap_enabled(self._remap_unlocked)

    def _on_anchor_dragged(self, which: int, value: float) -> None:
        if not (self._sensor_master and self.state.enabled and self._editable_two_point): return
        target = self.raw_a if int(which) == 0 else self.raw_b
        other = self.raw_b.value() if int(which) == 0 else self.raw_a.value()
        v = float(value)
        if abs(v - other) < 0.5:
            v = other - 1.0 if other > ADC_MIN else other + 1.0
        target.setValue(max(ADC_MIN, min(ADC_MAX, v)))

    def _on_editor_changed(self, unused_value=None) -> None:
        if self._block_ui or not self._editable_two_point: return
        try:
            self.state.set_points(self.raw_a.value(), self.angle_a.value(), self.raw_b.value(), self.angle_b.value())
        except CalibrationError:
            self.raw_a.setStyleSheet("QDoubleSpinBox { border: 1px solid #d08000; }")
            self.raw_b.setStyleSheet("QDoubleSpinBox { border: 1px solid #d08000; }")
            return
        self.raw_a.setStyleSheet(""); self.raw_b.setStyleSheet("")
        self.sync_bars(); self.changed.emit(self.state.name)

    def _reset_row(self) -> None:
        # Compatibility fallback. Normal operation is routed to the parent
        # controller through resetRequested so the authoritative session
        # reference is used even after Load/Save/rebuild operations.
        if not self._editable_two_point:
            return
        self.resetRequested.emit(self.state.name)

    def sync_editors_from_state(self) -> None:
        if not self._editable_two_point: return
        self._block_ui = True
        try:
            self.raw_a.setValue(self.state.raw_a); self.raw_b.setValue(self.state.raw_b)
            self.angle_a.setValue(self.state.angle_a); self.angle_b.setValue(self.state.angle_b)
            self.use_box.setChecked(bool(self.state.enabled)); self.auto_box.setChecked(bool(self.state.auto_enabled))
        finally:
            self._block_ui = False
        self.sync_bars()

    def sync_bars(self) -> None:
        if not self._editable_two_point: return
        self.raw_bar.set_raw(self.state.raw_current, self.state.raw_a, self.state.raw_b)
        angle = self.state.angle_current
        if angle is None and self.state.raw_current is not None:
            try: angle = self.state.local_angle()
            except CalibrationError: angle = None
        self.angle_bar.set_angle(angle, self.state.angle_a, self.state.angle_b)

    def refresh_live(self, raw: Optional[float], calibrated: Optional[float]) -> None:
        self.state.raw_current = raw; self.state.angle_current = calibrated
        self.raw_current.setText("—" if raw is None else "{:.0f}".format(float(raw)))
        angle = calibrated
        if angle is None and raw is not None:
            try: angle = self.state.local_angle(raw)
            except CalibrationError: angle = None
        self.angle_current.setText("—" if angle is None else "{:.1f}°".format(float(angle)))
        self.sync_bars()

class PostureCalibrationWidget(QtWidgets.QWidget):
    """Top-level posture calibration widget."""
    def save_settings(self, settings):
        return None

    def restore_settings(self, settings):
        return None


class RqtTitleBarPlaceholder(QtWidgets.QWidget):
    """Zero-height standalone-rqt title bar with the DockWidget settings API.

    qt_gui's DockWidget persists settings on its *title bar widget*.  Delta-1.4
    replaced that title bar with a plain QWidget to hide the duplicate heading,
    which caused the shutdown AttributeError seen on the user's machine.
    """
    def __init__(self, parent=None):
        super(RqtTitleBarPlaceholder, self).__init__(parent)
        self.setFixedHeight(0)

    def save_settings(self, settings):
        return None

    def restore_settings(self, settings):
        return None

# ---------------------------------------------------------------------------
# Main rqt plugin (class name intentionally unchanged)
# ---------------------------------------------------------------------------
class SrGuiCybergloveCalibrator(Plugin):
    """Data-driven free-motion posture calibrator for RBM gloves."""

    name = "RBM Posture Calibration"

    def __init__(self, context):
        super(SrGuiCybergloveCalibrator, self).__init__(context)
        self.setObjectName("RBM Posture Calibration")

        self.ns = _normalize_namespace(rospy.get_namespace())
        self.source_raw_topic = _topic(self.ns, SOURCE_RAW_TOPIC_SUFFIX)
        self.raw_topic = _topic(self.ns, RAW_TOPIC_SUFFIX)
        self.calibrated_topic = _topic(self.ns, CALIBRATED_TOPIC_SUFFIX)
        self._lock = threading.RLock()
        self._source_raw_values: List[float] = []
        self._source_raw_names: List[str] = []
        self._raw_values: Dict[str, float] = {}
        self._calibrated_values: Dict[str, float] = {}
        self._inventory: List[str] = []
        self._states: "OrderedDict[str, SensorCalibrationState]" = OrderedDict()
        self._rows: Dict[str, SensorRow] = {}
        self._selected_name: Optional[str] = None
        self._dirty = False
        self._session_modified = False
        self._shutdown = False
        self._last_apply_monotonic = 0.0
        self._last_apply_message = "not applied yet"
        self._pending_calibration_apply = False
        self._reload_retry_not_before = 0.0
        self._reload_retry_interval_s = 2.0
        self._last_tick_monotonic = time.monotonic()
        self._calibration_running = False
        self._working_base_calibration = None
        self._session_reference_calibration = None
        self._loaded_session_source = ""
        self._standalone_rqt = ("-s" in sys.argv and any("sr_gui_cyberglove_calibrator" in str(a) for a in sys.argv))
        self._tactile_count: Optional[int] = None
        self._imu_present: Optional[bool] = None
        self._serial_number = "—"
        self._product_number = "—"
        self._hand_map_dialog = None
        self._svg_hover_name = None
        self._expert_window_size = None
        self._column_resize_guard = False

        # Delta-1.12 channel->sensor canonicalisation state. The mapping is a
        # complete permutation: every physical input channel is assigned exactly
        # one physical sensor identity, then the canonicalizer republishes those
        # values in the legacy matrix order.
        self._canonicalizer_node = _topic(self.ns, CHANNEL_CANONICALIZER_NODE).rstrip("/")
        self.mapping_reload_service_name = self._canonicalizer_node + CHANNEL_RELOAD_SERVICE_SUFFIX
        self.mapping_param = self._canonicalizer_node + "/channel_to_sensor"
        self.canonical_order_param = self._canonicalizer_node + "/canonical_order"
        self.mapping_file_param = self._canonicalizer_node + "/mapping_file"
        self._canonical_order: List[str] = list(LEGACY_CANONICAL_ORDER_16)
        self._channel_map: List[str] = list(LEGACY_CANONICAL_ORDER_16)
        self._baseline_channel_map: List[str] = list(self._channel_map)
        self.channel_mapping_file: str = ""
        self._mapping_dirty = False
        self._loaded_mapping_source = ""
        self._mapping_available = False
        self._mapping_last_message = "canonicalizer not discovered"
        self._mapping_reload_proxy = None
        self._mapping_combo_block = False
        self._last_mapping_discovery_monotonic = 0.0

        self._signals = _GuiSignals()
        self._signals.inventory_changed.connect(self._rebuild_rows)
        self._signals.metadata_changed.connect(self._refresh_header)

        self.reload_service_name = self._discover_reload_service()
        self.calibration_param = self._discover_calibration_param()
        self.calibration_file = self._discover_calibration_file()
        self._reload_proxy = (
            rospy.ServiceProxy(self.reload_service_name, Trigger)
            if self.reload_service_name
            else None
        )
        self._discover_channel_mapping_runtime()

        self._baseline_calibration = self._load_current_calibration()
        self._calibration_order = self._load_preferred_order()
        self._session_start_calibration = clone_calibration(self._baseline_calibration)
        self._working_base_calibration = clone_calibration(self._baseline_calibration)
        self._session_reference_calibration = clone_calibration(self._baseline_calibration)

        self._settings = QtCore.QSettings("Robomantic", "PostureCalibratorDelta1")
        self._make_widget(context)
        self._widget.installEventFilter(self)
        self._make_shortcuts()

        self._source_raw_sub = rospy.Subscriber(
            self.source_raw_topic, JointState, self._source_raw_callback, queue_size=1
        )
        self._raw_sub = rospy.Subscriber(
            self.raw_topic, JointState, self._raw_callback, queue_size=1
        )
        self._cal_sub = rospy.Subscriber(
            self.calibrated_topic, JointState, self._calibrated_callback, queue_size=1
        )
        self._tactile_sub = self._try_subscribe_tactile()

        self._timer = QtCore.QTimer(self._widget)
        self._timer.setInterval(UI_TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # Device metadata discovery can involve service introspection. Run after
        # the GUI is shown so startup stays responsive.
        QtCore.QTimer.singleShot(50, self._discover_device_metadata)
        self._refresh_header()

    # ----- ROS discovery -------------------------------------------------
    def _discover_reload_service(self) -> Optional[str]:
        try:
            services = rosservice.get_service_list(namespace=self.ns)
        except Exception as exc:
            rospy.logwarn("Delta-1: cannot list services: %s", exc)
            return None
        candidates = [s for s in services if s.endswith(RELOAD_SERVICE_SUFFIX)]
        if not candidates:
            return None
        for service in candidates:
            if "glove_bend_sensor_calib" in service:
                return service
        return sorted(candidates)[0]

    def _discover_calibration_param(self) -> str:
        preferred = _topic(self.ns, CALIBRATION_PARAM_SUFFIX)
        if rospy.has_param(preferred):
            return preferred
        if rospy.has_param("/cyberglove/calibration"):
            return "/cyberglove/calibration"
        return preferred

    def _discover_calibration_file(self) -> str:
        candidates = []
        if self.reload_service_name:
            node_path = self.reload_service_name[: -len(RELOAD_SERVICE_SUFFIX)]
            candidates.append(node_path + "/path_to_calibration")
        candidates.extend([
            _topic(self.ns, "cyberglove/path_to_calibration"),
            "/cyberglove/path_to_calibration",
        ])
        for param in candidates:
            try:
                if rospy.has_param(param):
                    value = rospy.get_param(param)
                    if value:
                        return os.path.abspath(os.path.expanduser(str(value)))
            except Exception:
                pass
        return ""

    def _discover_channel_mapping_runtime(self) -> None:
        """Discover Delta-1.12 canonicalizer params/service without blocking startup."""
        try:
            if rospy.has_param(self.canonical_order_param):
                canonical = list(rospy.get_param(self.canonical_order_param))
            else:
                canonical = list(LEGACY_CANONICAL_ORDER_16)
            if rospy.has_param(self.mapping_param):
                mapping = list(rospy.get_param(self.mapping_param))
            else:
                mapping = list(canonical)
            validate_mapping(canonical, mapping, require_complete=True)
            self._canonical_order = canonical
            self._channel_map = mapping
            self._baseline_channel_map = list(mapping)
            if rospy.has_param(self.mapping_file_param):
                value = rospy.get_param(self.mapping_file_param)
                if value:
                    self.channel_mapping_file = os.path.abspath(os.path.expanduser(str(value)))
            services = rosservice.get_service_list(namespace=self.ns)
            self._mapping_available = self.mapping_reload_service_name in services
            if self._mapping_available:
                self._mapping_reload_proxy = rospy.ServiceProxy(
                    self.mapping_reload_service_name, Trigger
                )
                self._mapping_last_message = "mapping ready"
            else:
                self._mapping_last_message = "canonicalizer service not found"
        except Exception as exc:
            self._mapping_available = False
            self._mapping_last_message = "mapping discovery failed: {}".format(exc)
            rospy.logwarn("Delta-1.12 channel mapping unavailable: %s", exc)

    def _apply_channel_mapping(self, mapping: List[str], reason: str = "live") -> bool:
        """Apply a complete channel permutation to the canonicalizer live."""
        try:
            validate_mapping(self._canonical_order, mapping, require_complete=True)
            if not self._mapping_available or self._mapping_reload_proxy is None:
                self._mapping_last_message = "blocked: canonicalizer service missing"
                return False
            rospy.set_param(self.canonical_order_param, list(self._canonical_order))
            rospy.set_param(self.mapping_param, list(mapping))
            response = self._mapping_reload_proxy()
            if hasattr(response, "success") and not response.success:
                self._mapping_last_message = "reload rejected: {}".format(
                    getattr(response, "message", "unknown reason")
                )
                return False
            self._channel_map = list(mapping)
            self._mapping_last_message = "{} mapping applied".format(reason)
            return True
        except Exception as exc:
            self._mapping_last_message = "mapping apply failed: {}".format(exc)
            rospy.logerr("Delta-1.12 mapping apply failed: %s", exc)
            return False

    def _load_current_calibration(self) -> OrderedDict:
        try:
            if rospy.has_param(self.calibration_param):
                return calibration_from_ros_param(rospy.get_param(self.calibration_param))
        except Exception as exc:
            rospy.logwarn("Delta-1: cannot read calibration parameter %s: %s", self.calibration_param, exc)
        if self.calibration_file and os.path.isfile(self.calibration_file):
            try:
                return parse_calibration_xml(self.calibration_file)
            except CalibrationError as exc:
                rospy.logerr("Delta-1: %s", exc)
        return OrderedDict()

    def _load_preferred_order(self) -> List[str]:
        if self.calibration_file and os.path.isfile(self.calibration_file):
            try:
                return list(parse_calibration_xml(self.calibration_file).keys())
            except CalibrationError:
                pass
        return list(self._baseline_calibration.keys())

    def _try_subscribe_tactile(self):
        try:
            from tactile_msgs.msg import TactileState

            return rospy.Subscriber(
                _topic(self.ns, "TactileGlove"),
                TactileState,
                self._tactile_callback,
                queue_size=1,
            )
        except Exception:
            return None

    def _discover_device_metadata(self) -> None:
        # Serial number from the existing serial protocol service, if available.
        try:
            services = rosservice.get_service_list(namespace=self.ns)
            serial_services = [s for s in services if s.endswith("get_serialnumber")]
            if serial_services:
                from agni_serial_protocol.srv import GetSerialNumber

                proxy = rospy.ServiceProxy(sorted(serial_services)[0], GetSerialNumber)
                response = proxy()
                self._serial_number = str(response.serial_number) or "—"
        except Exception as exc:
            rospy.logdebug("Delta-1: serial number discovery unavailable: %s", exc)

        # If the serial service returns a firmware/build descriptor rather than
        # the device SN, recover the SN from the per-device calibration path.
        if not re.match(r"^\d{4}-\d{4}$", self._serial_number or ""):
            match = re.search(r"/config/([^/]+)/posture/", self.calibration_file or "")
            if match:
                self._serial_number = match.group(1)

        # Product number from the serial-number config folder. Avoid PyYAML here
        # so this GUI introduces no additional dependency beyond the existing stack.
        if self._serial_number != "—":
            try:
                root = rospkg.RosPack().get_path("rbm_launchfiles")
                settings = os.path.join(root, "config", self._serial_number, "settings.yaml")
                if os.path.isfile(settings):
                    with open(settings, "r", encoding="utf-8") as handle:
                        for line in handle:
                            match = re.match(r"^\s*pn\s*:\s*([^#]+)", line)
                            if match:
                                self._product_number = match.group(1).strip().strip('"\'')
                                break
            except Exception as exc:
                rospy.logdebug("Delta-1: product number discovery unavailable: %s", exc)

        try:
            published = rospy.get_published_topics(self.ns)
            self._imu_present = any(
                name.endswith("/BNO08X") or msg_type == "sensor_msgs/Imu"
                for name, msg_type in published
            )
        except Exception:
            self._imu_present = None

        self._signals.metadata_changed.emit()

    # ----- GUI construction ----------------------------------------------
    @staticmethod
    def _make_compact_button(text: str, tooltip: str, callback) -> QtWidgets.QToolButton:
        button = QtWidgets.QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.setFixedWidth(23)
        button.setFixedHeight(20)
        button.setStyleSheet("QToolButton { padding: 0px 1px; margin: 0px; }")
        button.clicked.connect(callback)
        return button

    def _make_widget(self, context) -> None:
        self._widget = PostureCalibrationWidget()
        self._widget.setObjectName("RBMPostureCalibration")
        self._widget.setWindowTitle("RBM Posture Calibration")
        try:
            posture_icon = QtGui.QIcon(resource_path(False))
            self._widget.setWindowIcon(posture_icon)
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.setWindowIcon(posture_icon)
        except Exception:
            pass
        self._widget.setMinimumWidth(330)
        if context is not None:
            context.add_widget(self._widget)
        if self._standalone_rqt:
            QtCore.QTimer.singleShot(0, self._hide_standalone_dock_titlebar)
            QtCore.QTimer.singleShot(150, self._hide_standalone_dock_titlebar)

        outer = QtWidgets.QVBoxLayout(self._widget)
        self._outer_layout = outer
        outer.setContentsMargins(5, 3, 5, 0)
        outer.setSpacing(1)

        # Compact top area: three left-aligned functional rows share one line
        # block with the always-visible hand map at the upper right.
        top_band = QtWidgets.QHBoxLayout()
        top_band.setContentsMargins(0, 0, 0, 0); top_band.setSpacing(4)
        top_left = QtWidgets.QWidget(); tl = QtWidgets.QVBoxLayout(top_left)
        tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(3)

        self.header_details = ElidedLabel()
        font = QtGui.QFont(self.header_details.font()); font.setBold(True); self.header_details.setFont(font)
        self.header_details.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        tl.addWidget(self.header_details, 0, QtCore.Qt.AlignTop)

        file_line = QtWidgets.QHBoxLayout(); file_line.setContentsMargins(0,0,0,0); file_line.setSpacing(1)
        active_label = QtWidgets.QLabel("Active:")
        active_palette = active_label.palette(); active_palette.setColor(QtGui.QPalette.WindowText, active_palette.color(QtGui.QPalette.Mid)); active_label.setPalette(active_palette)
        file_line.addWidget(active_label)
        self.path_label = SelectablePathLine(self.calibration_file or "active calibration path unknown"); file_line.addWidget(self.path_label, 1)
        self.reset_button = QtWidgets.QPushButton("Reset"); self.reset_button.setMinimumWidth(52); self.reset_button.clicked.connect(self._reset_session)
        self.load_button = QtWidgets.QPushButton("Load…"); self.load_button.setMinimumWidth(58); self.load_button.clicked.connect(self._load_session_calibration)
        self.save_button = QtWidgets.QPushButton("Save active"); self.save_button.setMinimumWidth(88); self.save_button.clicked.connect(self._save_active)
        self.export_button = QtWidgets.QPushButton("Export…"); self.export_button.setMinimumWidth(74); self.export_button.clicked.connect(self._export_copy)
        for b in (self.reset_button,self.load_button,self.save_button,self.export_button): file_line.addWidget(b)
        tl.addLayout(file_line)

        control_line = QtWidgets.QHBoxLayout(); control_line.setContentsMargins(0,0,0,0); control_line.setSpacing(3)
        self.run_button = QtWidgets.QPushButton("Start Calib"); self.run_button.setCheckable(True); self.run_button.setMinimumWidth(92); self.run_button.setMinimumHeight(36)
        self.run_button.setToolTip("Start/freeze free-motion calibration (Space)"); self.run_button.toggled.connect(self._set_calibration_running); control_line.addWidget(self.run_button)
        control_line.addSpacing(5)
        auto_block = QtWidgets.QWidget(); auto_layout = QtWidgets.QHBoxLayout(auto_block); auto_layout.setContentsMargins(0,0,0,0); auto_layout.setSpacing(2)
        self.narrow_enabled = QtWidgets.QCheckBox("Auto-narrow"); self.narrow_enabled.setChecked(True); self.narrow_enabled.toggled.connect(self._refresh_hierarchy_ui)
        self.narrow_rate = QtWidgets.QDoubleSpinBox(); self.narrow_rate.setRange(0.0,500.0); self.narrow_rate.setDecimals(1); self.narrow_rate.setValue(30.0); self.narrow_rate.setSuffix(" ADC/s"); self.narrow_rate.setMaximumWidth(92); self.narrow_rate.setLocale(QtCore.QLocale.c())
        self.collapse_button = QtWidgets.QPushButton("Collapse"); self.collapse_button.clicked.connect(self._collapse_auto_rows)
        auto_layout.addWidget(self.narrow_enabled); auto_layout.addWidget(self.narrow_rate); auto_layout.addWidget(self.collapse_button)
        control_line.addWidget(auto_block); control_line.addSpacing(5)
        self.expert_toggle = QtWidgets.QCheckBox("Expert"); self.expert_toggle.setChecked(True); self.expert_toggle.toggled.connect(self._set_expert_visible); control_line.addWidget(self.expert_toggle); control_line.addStretch(1)
        tl.addLayout(control_line)
        tl.addStretch(1)
        top_band.addWidget(top_left, 1, QtCore.Qt.AlignTop)

        # Hand map is ~25% larger than Delta-1.6 and its tools overlay its wrist
        # region, consuming no additional vertical row.
        map_compact = QtWidgets.QWidget(); map_compact.setFixedSize(108, 118)
        map_grid = QtWidgets.QGridLayout(map_compact); map_grid.setContentsMargins(0,0,0,0); map_grid.setSpacing(0)
        mirror = self.ns.strip("/").lower() == "right"
        self.hand_xray_box = QtWidgets.QCheckBox("X-ray"); self.hand_xray_box.setChecked(bool(self._settings.value("handmap/xray", False, type=bool)))
        self.hand_map_small = HandSvgWidget(map_compact, mirror=mirror, xray=self.hand_xray_box.isChecked()); self.hand_map_small.setMinimumSize(108,118); self.hand_map_small.setMaximumSize(108,118)
        map_grid.addWidget(self.hand_map_small,0,0)
        map_tools = QtWidgets.QWidget(map_compact); mt=QtWidgets.QHBoxLayout(map_tools); mt.setContentsMargins(1,0,1,1); mt.setSpacing(1); mt.addWidget(self.hand_xray_box); mt.addStretch(1)
        self.hand_map_open=QtWidgets.QToolButton(); self.hand_map_open.setText("↗"); self.hand_map_open.setToolTip("Open/close large hand/sensor map"); mt.addWidget(self.hand_map_open)
        map_tools.setStyleSheet("background: rgba(245,245,245,185);")
        map_grid.addWidget(map_tools,0,0,alignment=QtCore.Qt.AlignBottom)
        top_band.addWidget(map_compact, 0, QtCore.Qt.AlignTop | QtCore.Qt.AlignRight)
        outer.addLayout(top_band)

        self.hand_xray_box.toggled.connect(self._set_hand_map_xray)
        self.hand_map_open.clicked.connect(self._open_hand_map_dialog)
        self.hand_map_small.doubleClicked.connect(self._open_hand_map_dialog)
        self.hand_map_small.sensorClicked.connect(lambda name: self._select_sensor(name, scroll=True))
        self.hand_map_small.sensorHovered.connect(self._on_svg_sensor_hovered)

        # Everything below is Expert-only: method parameters, mapping persistence,
        # and the single fused channel/sensor/calibration table.
        self.expert_container = QtWidgets.QWidget()
        expert = QtWidgets.QVBoxLayout(self.expert_container); expert.setContentsMargins(0,0,0,0); expert.setSpacing(2)

        self.proto_group = QtWidgets.QGroupBox("Expert parameters")
        pl = QtWidgets.QHBoxLayout(self.proto_group); pl.setContentsMargins(5,2,5,2); pl.setSpacing(4)
        self.persistence_ms=QtWidgets.QSpinBox(); self.persistence_ms.setRange(0,1000); self.persistence_ms.setValue(DEFAULT_PERSISTENCE_MS); self.persistence_ms.setSuffix(" ms")
        self.apply_hz=QtWidgets.QDoubleSpinBox(); self.apply_hz.setRange(0.2,10.0); self.apply_hz.setDecimals(1); self.apply_hz.setValue(DEFAULT_APPLY_HZ); self.apply_hz.setSuffix(" Hz"); self.apply_hz.setLocale(QtCore.QLocale.c())
        pl.addWidget(QtWidgets.QLabel("Persistence")); pl.addWidget(self.persistence_ms); pl.addWidget(QtWidgets.QLabel("Live apply")); pl.addWidget(self.apply_hz)
        self.columns_button=QtWidgets.QToolButton(); self.columns_button.setText("Columns"); self.columns_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu=QtWidgets.QMenu(self.columns_button)
        self.col_calibration_action=menu.addAction("Calibration bars"); self.col_calibration_action.setCheckable(True); self.col_calibration_action.setChecked(True)
        self.col_current_action=menu.addAction("Current values"); self.col_current_action.setCheckable(True); self.col_current_action.setChecked(True)
        self.col_exact_action=menu.addAction("Exact min/max"); self.col_exact_action.setCheckable(True); self.col_exact_action.setChecked(True)
        for action in (self.col_calibration_action,self.col_current_action,self.col_exact_action): action.toggled.connect(self._apply_column_visibility)
        self.columns_button.setMenu(menu); pl.addWidget(self.columns_button); pl.addStretch(1)
        expert.addWidget(self.proto_group)

        # One real QTableWidget owns header + data geometry.  This prevents
        # live-value width changes from shifting the bar column and guarantees
        # that column hiding collapses all of its space.
        self.calibration_table = QtWidgets.QTableWidget(1, 6)
        self.calibration_table.setObjectName("PostureCalibrationTable")
        self.calibration_table.horizontalHeader().setVisible(False)
        self.calibration_table.verticalHeader().setVisible(False)
        self.calibration_table.setShowGrid(True)
        self.calibration_table.setAlternatingRowColors(True)
        # Row highlighting is painted explicitly by SensorRow. Native table
        # selection used the same blue as calibration flags and also interfered
        # with raw-anchor drag gestures.
        self.calibration_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.calibration_table.setMouseTracking(True)
        self.calibration_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.calibration_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.calibration_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.calibration_table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.calibration_table.setMinimumHeight(180)
        self.calibration_table.setStyleSheet(
            "QTableWidget#PostureCalibrationTable { gridline-color: palette(mid); }"
            "QTableWidget#PostureCalibrationTable::item { padding: 0px; }"
        )
        hh = self.calibration_table.horizontalHeader()
        # All sections are user-resizable, but each has a semantic minimum.
        # If the window becomes narrower, a horizontal scrollbar is preferable
        # to cropping/overlapping the calibration bars.
        for col in range(6):
            hh.setSectionResizeMode(col, QtWidgets.QHeaderView.Interactive)
        self._column_min_widths = {
            0: SensorRow.CHANNEL_W,
            1: SensorRow.SENSOR_W,
            2: SensorRow.CALIBRATION_W,
            3: SensorRow.CURRENT_W,
            4: SensorRow.EXACT_W,
            5: SensorRow.AUTO_W,
        }
        for col, width in self._column_min_widths.items():
            try:
                saved = int(self._settings.value("table/column_{}".format(col), width))
            except Exception:
                saved = width
            self.calibration_table.setColumnWidth(col, max(width, saved))
        hh.sectionResized.connect(self._on_table_section_resized)
        self.calibration_table.setRowHeight(0, 48)
        self.calibration_table.cellEntered.connect(self._on_table_cell_entered)
        self.calibration_table.cellClicked.connect(self._on_table_cell_clicked)
        self._calibration_viewport = self.calibration_table.viewport()
        self._calibration_viewport.installEventFilter(self)

        def _header_host():
            w=QtWidgets.QWidget(); w.setAutoFillBackground(False)
            return w

        ch_head=_header_host(); chl=QtWidgets.QVBoxLayout(ch_head); chl.setContentsMargins(3,1,3,1); chl.setSpacing(0)
        self.bend_enable=QtWidgets.QCheckBox("Ch."); self.bend_enable.setChecked(True); self.bend_enable.toggled.connect(self._refresh_hierarchy_ui); chl.addWidget(self.bend_enable,alignment=QtCore.Qt.AlignLeft)
        ch_buttons=QtWidgets.QHBoxLayout(); ch_buttons.setContentsMargins(0,0,0,0); ch_buttons.setSpacing(1)
        self.sensor_all=self._make_compact_button("∀", "Enable all channels", lambda: self._set_all_sensors(True)); ch_buttons.addWidget(self.sensor_all)
        self.sensor_none=self._make_compact_button("∄", "Disable all channels", lambda: self._set_all_sensors(False)); ch_buttons.addWidget(self.sensor_none); chl.addLayout(ch_buttons)
        self.calibration_table.setCellWidget(0,0,ch_head)

        sensor_head=_header_host(); sl=QtWidgets.QVBoxLayout(sensor_head); sl.setContentsMargins(3,1,3,1); sl.setSpacing(0)
        sensor_label=QtWidgets.QLabel("Sensor"); sensor_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter); sl.addWidget(sensor_label)
        sl.addStretch(1)
        self.calibration_table.setCellWidget(0,1,sensor_head)

        def secondary_header(text, tooltip=""):
            label=QtWidgets.QLabel(text); label.setAlignment(QtCore.Qt.AlignCenter)
            palette=label.palette(); palette.setColor(QtGui.QPalette.WindowText,palette.color(QtGui.QPalette.Mid)); label.setPalette(palette)
            if tooltip: label.setToolTip(tooltip)
            return label
        self.calibration_header=secondary_header("Raw ADC 0 ─── 4095\nAngle")
        self.current_header=secondary_header("Current\nraw / angle")
        self.exact_header=secondary_header("min        max\nmin°      max°")
        self.calibration_table.setCellWidget(0,2,self.calibration_header)
        self.calibration_table.setCellWidget(0,3,self.current_header)
        self.calibration_table.setCellWidget(0,4,self.exact_header)

        auto_head=_header_host(); ah=QtWidgets.QVBoxLayout(auto_head); ah.setContentsMargins(2,1,2,1); ah.setSpacing(0)
        self.global_auto=QtWidgets.QCheckBox("Auto"); self.global_auto.setChecked(True); self.global_auto.toggled.connect(self._refresh_hierarchy_ui); ah.addWidget(self.global_auto,alignment=QtCore.Qt.AlignHCenter)
        ab=QtWidgets.QHBoxLayout(); ab.setContentsMargins(0,0,0,0); ab.setSpacing(1)
        self.auto_all_button=self._make_compact_button("∀", "Enable Auto for all enabled channels", lambda: self._set_all_auto(True)); ab.addWidget(self.auto_all_button)
        self.auto_none_button=self._make_compact_button("∄", "Disable Auto for all channels", lambda: self._set_all_auto(False)); ab.addWidget(self.auto_none_button); ah.addLayout(ab)
        self.calibration_table.setCellWidget(0,5,auto_head)

        for col in range(6):
            item=QtWidgets.QTableWidgetItem(); item.setFlags(QtCore.Qt.ItemIsEnabled); self.calibration_table.setItem(0,col,item)
            item.setBackground(self.calibration_table.palette().color(QtGui.QPalette.AlternateBase))
        expert.addWidget(self.calibration_table,1)

        # Mapping persistence is intentionally below the table and remains
        # locked during normal calibration.  The table shows plain sensor names
        # until Remap is explicitly enabled.
        self.mapping_controls = QtWidgets.QWidget()
        self.mapping_controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        mapping_line=QtWidgets.QHBoxLayout(self.mapping_controls); mapping_line.setContentsMargins(0,0,0,0); mapping_line.setSpacing(2)
        self.mapping_status=SelectablePathLine(); mapping_line.addWidget(self.mapping_status,1)
        self.remap_unlock=QtWidgets.QCheckBox("Remap"); self.remap_unlock.setChecked(False); self.remap_unlock.setToolTip("Unlock channel→sensor dropdowns. Disabled by default to prevent accidental remapping."); self.remap_unlock.toggled.connect(self._set_remap_unlocked); mapping_line.addWidget(self.remap_unlock)
        self.mapping_load_button=QtWidgets.QPushButton("Load map…"); self.mapping_load_button.clicked.connect(self._load_channel_mapping); mapping_line.addWidget(self.mapping_load_button)
        self.mapping_revert_button=QtWidgets.QPushButton("Revert mapping"); self.mapping_revert_button.clicked.connect(self._revert_channel_mapping); mapping_line.addWidget(self.mapping_revert_button)
        self.mapping_save_button=QtWidgets.QPushButton("Save mapping"); self.mapping_save_button.clicked.connect(self._save_channel_mapping); mapping_line.addWidget(self.mapping_save_button)
        # A vertical splitter separates the table from the lower information
        # area. The handle sits directly below the table in Expert mode and
        # lets users trade table height for Help/Debug height without changing
        # any calibration geometry. Mapping persistence lives in the lower pane
        # but remains Expert-only.
        self.lower_container = QtWidgets.QWidget()
        self.lower_container.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        lower = QtWidgets.QVBoxLayout(self.lower_container)
        lower.setContentsMargins(0,0,0,0); lower.setSpacing(1); lower.setAlignment(QtCore.Qt.AlignTop)
        lower.addWidget(self.mapping_controls, 0, QtCore.Qt.AlignTop)

        self.bottom_controls = QtWidgets.QWidget()
        self.bottom_controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        bottom=QtWidgets.QHBoxLayout(self.bottom_controls); bottom.setContentsMargins(0,0,0,0); bottom.setSpacing(2)
        self.instructions_toggle=QtWidgets.QToolButton(); self.instructions_toggle.setText("Help / keys"); self.instructions_toggle.setCheckable(True); self.instructions_toggle.toggled.connect(self._toggle_instructions); bottom.addWidget(self.instructions_toggle)
        self.debug_toggle=QtWidgets.QToolButton(); self.debug_toggle.setText("Debug / architecture"); self.debug_toggle.setCheckable(True); self.debug_toggle.toggled.connect(self._toggle_debug_panel); bottom.addWidget(self.debug_toggle); bottom.addStretch(1)
        lower.addWidget(self.bottom_controls, 0, QtCore.Qt.AlignTop)
        self.instructions_panel=QtWidgets.QLabel("Upper bar: raw ADC; blue flags are calibration anchors and can be dragged.\nLower bar: calibrated angle; orange extension means out-of-range. Space: Start/Freeze · ↑/↓ select sensor · Enter toggle selected Auto · Ctrl+S Save · Ctrl+R Reset · F12 Debug")
        self.instructions_panel.setWordWrap(True); self.instructions_panel.setVisible(False); self.instructions_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed); lower.addWidget(self.instructions_panel, 0, QtCore.Qt.AlignTop)
        self.debug_panel=QtWidgets.QPlainTextEdit(); self.debug_panel.setReadOnly(True); self.debug_panel.setMinimumHeight(80); self.debug_panel.setVisible(False); self.debug_panel.setFont(_monospace_font(self.debug_panel.font())); self.debug_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding); lower.addWidget(self.debug_panel, 1)
        self.status_label=ElidedLabel(); self.status_label.setFixedHeight(self.status_label.fontMetrics().height()+1); lower.addWidget(self.status_label, 0, QtCore.Qt.AlignTop)

        self.content_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.content_splitter.setObjectName("PostureCalibrationVerticalSplitter")
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(6)
        self.content_splitter.addWidget(self.expert_container)
        self.content_splitter.addWidget(self.lower_container)
        self.content_splitter.setStretchFactor(0, 4)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([520, 110])
        self.content_splitter.splitterMoved.connect(self._remember_content_splitter_sizes)
        outer.addWidget(self.content_splitter, 1)

        self._set_expert_visible(True)
        self._restore_content_splitter_sizes()
        self._apply_column_visibility()
        self._refresh_hierarchy_ui()

    def _hide_standalone_dock_titlebar(self) -> None:
        """Hide rqt's duplicate dock heading in standalone -s mode."""
        if not self._standalone_rqt or self._widget is None:
            return
        parent = self._widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QtWidgets.QDockWidget):
                try:
                    blank = RqtTitleBarPlaceholder(parent)
                    parent.setTitleBarWidget(blank)
                    self._dock_title_placeholder = blank
                except Exception:
                    pass
                return
            parent = parent.parentWidget()

    def _make_shortcuts(self) -> None:
        self._shortcuts = []
        bindings = [
            (KEY_TOGGLE_CALIB, self._toggle_calibration_running),
            (KEY_PREVIOUS_SENSOR, lambda: self._move_selection(-1)),
            (KEY_NEXT_SENSOR, lambda: self._move_selection(1)),
            (KEY_TOGGLE_SENSOR_AUTO, self._toggle_selected_auto),
            (KEY_SAVE_ACTIVE, self._save_active),
            (KEY_RESET_SESSION, self._reset_session),
            (KEY_TOGGLE_DEBUG, self._toggle_debug_shortcut),
        ]
        for key, callback in bindings:
            shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(key), self._widget)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    # ----- callbacks / inventory -----------------------------------------
    def _source_raw_callback(self, msg: JointState) -> None:
        with self._lock:
            self._source_raw_names = list(msg.name)
            self._source_raw_values = [float(x) for x in msg.position]

    def _raw_callback(self, msg: JointState) -> None:
        values = {name: float(msg.position[i]) for i, name in enumerate(msg.name) if i < len(msg.position)}
        names = list(msg.name)
        inventory_changed = False
        with self._lock:
            self._raw_values = values
            if names != self._inventory:
                self._inventory = names
                inventory_changed = True
        if inventory_changed:
            self._signals.inventory_changed.emit(names)

    def _calibrated_callback(self, msg: JointState) -> None:
        values = {name: float(msg.position[i]) for i, name in enumerate(msg.name) if i < len(msg.position)}
        with self._lock:
            self._calibrated_values = values

    def _tactile_callback(self, msg) -> None:
        try:
            count = sum(len(sensor.values) for sensor in msg.sensors)
        except Exception:
            return
        if count != self._tactile_count:
            self._tactile_count = count
            self._signals.metadata_changed.emit()

    def _rebuild_rows(self, names: List[str]) -> None:
        if self._shutdown or not hasattr(self, "calibration_table"):
            return
        old_states = self._states
        self._rows = {}
        self._states = OrderedDict()

        for name in names:
            if name in old_states:
                state = old_states[name]
            else:
                points = self._working_base_calibration.get(name, [])
                state = SensorCalibrationState.from_points(
                    name, points,
                    enabled=self._settings.value(self.ns + "enabled/" + name, bool(points), type=bool),
                    auto_enabled=self._settings.value(self.ns + "auto/" + name, True, type=bool),
                )
            self._states[name] = state

        ordered_sensors = list(self._channel_map) if self._channel_map else list(names)
        self.calibration_table.setUpdatesEnabled(False)
        try:
            self.calibration_table.clearSelection()
            self.calibration_table.setRowCount(1 + len(ordered_sensors))
            remap_unlocked = bool(getattr(self, "remap_unlock", None) and self.remap_unlock.isChecked())
            for channel, name in enumerate(ordered_sensors):
                if name not in self._states:
                    points = self._working_base_calibration.get(name, [])
                    self._states[name] = SensorCalibrationState.from_points(name, points, enabled=bool(points), auto_enabled=True)
                state = self._states[name]
                baseline_points = self._working_base_calibration.get(name, state.to_points())
                row = SensorRow(
                    channel, state, baseline_points, self._canonical_order,
                    mapping_available=self._mapping_available,
                    remap_unlocked=remap_unlocked, parent=self.calibration_table,
                )
                row.set_alternate(bool(channel % 2))
                row.set_master_states(bool(self.bend_enable.isChecked()), bool(self.global_auto.isChecked()))
                row.changed.connect(self._on_row_changed)
                row.selected.connect(self._select_sensor)
                row.hovered.connect(self._on_sensor_row_hovered)
                row.mappingRequested.connect(self._on_channel_assignment_changed)
                row.mappingPopupOpened.connect(self._on_mapping_combo_opened)
                row.mappingCandidateHovered.connect(self._on_mapping_candidate_hovered)
                row.resetRequested.connect(self._reset_sensor_to_reference)
                if channel < len(self._source_raw_names):
                    row.set_source_label(self._source_raw_names[channel])
                self._rows[name] = row
                row.install_into_table(self.calibration_table, channel + 1)
        finally:
            self.calibration_table.setUpdatesEnabled(True)

        self._apply_column_visibility()
        if ordered_sensors:
            if self._selected_name not in self._rows:
                self._selected_name = ordered_sensors[0]
            self._select_sensor(self._selected_name)
        self._refresh_hierarchy_ui(); self._refresh_header(); self._refresh_mapping_status(); self._refresh_debug()

    def _on_row_changed(self, name: str) -> None:
        if name in self._states:
            state = self._states[name]
            self._settings.setValue(self.ns + "enabled/" + name, bool(state.enabled))
            self._settings.setValue(self.ns + "auto/" + name, bool(state.auto_enabled))
        self._dirty = True; self._session_modified = True
        self._refresh_hierarchy_ui(); self._refresh_header()

    # ----- Delta-1.12 fused channel mapping --------------------------------
    def _rebuild_mapping_table(self) -> None:
        # Backward-compatible internal hook: the fused QTableWidget is the mapping table.
        if hasattr(self, "calibration_table"):
            self._rebuild_rows(list(self._inventory))

    def _refresh_mapping_live_cells(self) -> None:
        with self._lock:
            source_names = list(self._source_raw_names)
        for channel, sensor in enumerate(self._channel_map):
            row = self._rows.get(sensor)
            if row is not None:
                row.set_mapping_available(self._mapping_available)
                row.set_remap_enabled(bool(getattr(self, "remap_unlock", None) and self.remap_unlock.isChecked()))
                row.set_source_label(source_names[channel] if channel < len(source_names) else "—")

    def _set_remap_unlocked(self, unlocked: bool) -> None:
        for row in self._rows.values():
            row.set_remap_enabled(bool(unlocked))

    def _on_mapping_combo_opened(self, channel: int, sensor) -> None:
        if 0 <= int(channel) < len(self._channel_map):
            current = self._channel_map[int(channel)]
            if current in self._rows:
                self._select_sensor(current)

    def _on_mapping_candidate_hovered(self, sensor) -> None:
        candidate = str(sensor) if sensor else None
        if hasattr(self, "hand_map_small"):
            self.hand_map_small.set_hovered_sensor(candidate)
        if self._hand_map_dialog is not None:
            self._hand_map_dialog.set_hovered_sensor(candidate)
        self._on_svg_sensor_hovered(candidate)

    def _on_channel_assignment_changed(self, channel: int, sensor: str) -> None:
        if not (getattr(self, "remap_unlock", None) and self.remap_unlock.isChecked()):
            return
        old = list(self._channel_map)
        try:
            candidate = swap_assignment(old, channel, str(sensor))
            if candidate == old:
                return
            if not self._apply_channel_mapping(candidate, reason="live"):
                raise ChannelMappingError(self._mapping_last_message)
            self._mapping_dirty = candidate != self._baseline_channel_map
            self._refresh_header()
            self._rebuild_rows(list(self._inventory))
        except Exception as exc:
            self._channel_map = old
            self._mapping_last_message = "swap rejected: {}".format(exc)
            self._rebuild_rows(list(self._inventory))
            rospy.logwarn("Delta-1.12 channel swap rejected: %s", exc)

    def _load_channel_mapping(self) -> None:
        start = self.channel_mapping_file or (os.path.dirname(self.calibration_file) if self.calibration_file else os.path.expanduser("~"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._widget,
            "Load channel mapping into live session",
            start,
            "Channel mapping YAML (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        try:
            canonical, mapping, metadata, existed = load_mapping_file(path, fallback_canonical=self._canonical_order)
            if not existed:
                raise ChannelMappingError("selected mapping file does not exist")
            if list(canonical) != list(self._canonical_order):
                raise ChannelMappingError(
                    "mapping canonical order does not match the running posture chain"
                )
            if not self._apply_channel_mapping(list(mapping), reason="loaded"):
                raise ChannelMappingError(self._mapping_last_message)
            self._channel_map = list(mapping)
            self._mapping_dirty = self._channel_map != self._baseline_channel_map
            self._loaded_mapping_source = os.path.abspath(os.path.expanduser(path))
            self._mapping_last_message = "mapping loaded into live session"
            self._rebuild_rows(list(self._inventory))
            self._refresh_mapping_status()
        except Exception as exc:
            rospy.logerr("Delta-1.12 channel mapping load failed: %s", exc)
            QtWidgets.QMessageBox.critical(self._widget, "Mapping load failed", str(exc))

    def _save_channel_mapping(self) -> None:
        if not self._mapping_available:
            QtWidgets.QMessageBox.warning(
                self._widget,
                "Channel canonicalizer unavailable",
                "The Delta-1.12 canonicalizer service is not running. Restart the glove suite with the supplied processing.launch before saving a channel map.",
            )
            return
        target = self.channel_mapping_file
        if not target and self.calibration_file:
            target = os.path.join(os.path.dirname(self.calibration_file), "channel_to_sensor.yaml")
        if not target:
            QtWidgets.QMessageBox.warning(
                self._widget,
                "Channel map path unknown",
                "No mapping_file was exposed by the canonicalizer and the calibration path is unknown. Mapping will not be written.",
            )
            return
        try:
            result = safe_write_mapping(
                target,
                self._canonical_order,
                self._channel_map,
                metadata={
                    "tool": "RBM Posture Calibrator Delta-1.12",
                    "version": PLUGIN_VERSION,
                    "namespace": self.ns,
                    "serial_number": self._serial_number,
                    "product_number": self._product_number,
                    "semantics": "physical ADC channel (zero-based) -> physical sensor identity -> legacy canonical order",
                },
            )
            self.channel_mapping_file = result.get("target") or target
            try:
                rospy.set_param(self.mapping_file_param, self.channel_mapping_file)
            except Exception:
                pass
            self._baseline_channel_map = list(self._channel_map)
            self._mapping_dirty = False
            self._mapping_last_message = "mapping saved"
            self._refresh_mapping_status()
            QtWidgets.QMessageBox.information(
                self._widget,
                "Channel mapping saved safely",
                "Active mapping:\n{}\n\nBackup:\n{}\n\nAudit log:\n{}".format(
                    result.get("target"),
                    result.get("backup") or "(no previous mapping file existed)",
                    result.get("history") or "—",
                ),
            )
        except Exception as exc:
            rospy.logerr("Delta-1.12 channel mapping save failed: %s", exc)
            QtWidgets.QMessageBox.critical(self._widget, "Mapping save failed", str(exc))

    def _revert_channel_mapping(self) -> None:
        candidate = list(self._baseline_channel_map)
        if self._apply_channel_mapping(candidate, reason="revert"):
            self._channel_map = candidate
            self._mapping_dirty = False
            self._rebuild_mapping_table()

    def _refresh_mapping_status(self) -> None:
        if not hasattr(self, "mapping_status"):
            return
        state = "UNSAVED" if self._mapping_dirty else "saved/baseline"
        availability = "live" if self._mapping_available else "canonicalizer unavailable"
        path = self.channel_mapping_file or "mapping file not created yet"
        self.mapping_status.setFullText(path)
        loaded = (" · loaded " + os.path.basename(self._loaded_mapping_source)) if self._loaded_mapping_source else ""
        self.mapping_status.setToolTip("{} · {} · {}{}\n{}".format(state, availability, self._mapping_last_message, loaded, path))
        self.mapping_load_button.setEnabled(bool(self._mapping_available))
        self.mapping_save_button.setEnabled(bool(self._mapping_available))
        self.mapping_revert_button.setEnabled(bool(self._mapping_available and self._mapping_dirty))

    # ----- periodic processing -------------------------------------------
    def _auto_config(self) -> AutoRangeConfig:
        return AutoRangeConfig(
            persistence_s=float(self.persistence_ms.value()) / 1000.0,
            expansion_epsilon_adc=0.5,
            contraction_enabled=bool(self.narrow_enabled.isChecked()),
            contraction_rate_adc_s=float(self.narrow_rate.value()),
            min_span_adc=DEFAULT_MIN_SPAN_ADC,
        )

    def _tick(self) -> None:
        if self._shutdown:
            return
        now = time.monotonic()
        if (not self._mapping_available) and (now - self._last_mapping_discovery_monotonic) >= 2.0:
            self._last_mapping_discovery_monotonic = now
            was_available = self._mapping_available
            self._discover_channel_mapping_runtime()
            if self._mapping_available and not was_available:
                self._rebuild_mapping_table()
        with self._lock:
            raw = dict(self._raw_values)
            calibrated = dict(self._calibrated_values)

        self._refresh_mapping_live_cells()
        self._refresh_mapping_status()

        global_auto = bool(self._calibration_running and self.bend_enable.isChecked() and self.global_auto.isChecked())
        config = self._auto_config()
        for name, row in self._rows.items():
            raw_value = raw.get(name)
            calibrated_value = calibrated.get(name)
            row.refresh_live(raw_value, calibrated_value)
            state = self._states[name]
            if global_auto and raw_value is not None:
                changed = state.update_auto(raw_value, now, config)
                if changed:
                    row.sync_editors_from_state()
                    self._dirty = True
                    self._session_modified = True

        apply_interval = 1.0 / max(0.2, float(self.apply_hz.value()))
        if (self._dirty or self._pending_calibration_apply) and (now - self._last_apply_monotonic) >= apply_interval:
            if (not self._pending_calibration_apply) or now >= self._reload_retry_not_before:
                self._apply_live_calibration()

        self._last_tick_monotonic = now
        self._refresh_status()

    def _on_table_cell_entered(self, row: int, column: int) -> None:
        if row <= 0:
            self._on_sensor_row_hovered(None)
            return
        channel = row - 1
        if 0 <= channel < len(self._channel_map):
            sensor = self._channel_map[channel]
            if sensor in self._rows:
                self._on_sensor_row_hovered(sensor)

    def _on_table_cell_clicked(self, row: int, column: int) -> None:
        if row <= 0:
            return
        channel = row - 1
        if 0 <= channel < len(self._channel_map):
            sensor = self._channel_map[channel]
            if sensor in self._rows and sensor != self._selected_name:
                self._select_sensor(sensor)

    # ----- ROS live calibration ------------------------------------------
    def _working_calibration(self) -> OrderedDict:
        return merge_live_states(self._working_base_calibration, self._states)

    def _rediscover_reload_service(self) -> bool:
        """Refresh the bend-calibration reload service/proxy after node restarts.

        ROS service proxies can become stale when the processing node is restarted.
        File loading must remain valid even while that service is temporarily absent.
        """
        discovered = self._discover_reload_service()
        if not discovered:
            self.reload_service_name = None
            self._reload_proxy = None
            return False
        if discovered != self.reload_service_name or self._reload_proxy is None:
            self.reload_service_name = discovered
            self._reload_proxy = rospy.ServiceProxy(discovered, Trigger)
        return True

    def _mark_reload_pending(self, message: str) -> None:
        self._pending_calibration_apply = True
        self._reload_retry_not_before = time.monotonic() + self._reload_retry_interval_s
        self._last_apply_message = message

    def _apply_calibration_map(self, calibration: OrderedDict, reason: str = "live") -> bool:
        errors = validate_calibration_map(calibration)
        if errors:
            self._pending_calibration_apply = False
            self._last_apply_message = "blocked: " + errors[0]
            rospy.logwarn("Delta-1 calibration blocked: %s", "; ".join(errors))
            return False
        try:
            # Always update the parameter first.  A temporarily unavailable reload
            # service must not invalidate a successfully parsed/loaded calibration.
            rospy.set_param(self.calibration_param, calibration_to_ros_param(calibration))
        except Exception as exc:
            self._pending_calibration_apply = False
            self._last_apply_message = "parameter update failed: {}".format(exc)
            rospy.logerr("Delta-1 parameter update failed: %s", exc)
            return False

        # A proxy may be stale after a processing-node restart.  Try the current
        # proxy once, rediscover once on failure, then leave the apply pending.
        last_exc = None
        for attempt in range(2):
            if self._reload_proxy is None and not self._rediscover_reload_service():
                break
            try:
                response = self._reload_proxy()
                if hasattr(response, "success") and not response.success:
                    self._pending_calibration_apply = False
                    self._last_apply_message = "reload service reported failure"
                    return False
                self._pending_calibration_apply = False
                self._reload_retry_not_before = 0.0
                self._last_apply_message = "{} applied".format(reason)
                return True
            except Exception as exc:
                last_exc = exc
                self._reload_proxy = None
                if attempt == 0:
                    self._rediscover_reload_service()

        detail = "reload service temporarily unavailable"
        if last_exc is not None:
            detail += ": {}".format(last_exc)
        self._mark_reload_pending("{}; live apply pending".format(reason))
        # Throttle the diagnostic: pending apply is a recoverable runtime state,
        # not a reason to flood the terminal every UI tick.
        try:
            rospy.logwarn_throttle(5.0, "Delta-1.12 %s", detail)
        except Exception:
            pass
        return False

    def _apply_live_calibration(self) -> None:
        now = time.monotonic()
        if self._pending_calibration_apply and now < self._reload_retry_not_before:
            return
        calibration = self._working_calibration()
        if self._apply_calibration_map(calibration, reason="live"):
            self._dirty = False
        self._last_apply_monotonic = now
        self._refresh_debug()

    # ----- selection / keyboard ------------------------------------------
    def _select_sensor(self, name: str, scroll: bool = False) -> None:
        """Select a sensor without surprising viewport jumps.

        ``scroll`` is deliberately source-aware: keyboard navigation and SVG
        clicks request centring; ordinary clicks inside the calibration table
        (including flag drags and reset buttons) only update the selection/highlight.
        """
        if name not in self._rows:
            return
        self._selected_name = name
        for joint_name, row in self._rows.items():
            row.set_selected(joint_name == name)
        if hasattr(self, "hand_map_small"):
            self.hand_map_small.set_selected_sensor(name)
        if self._hand_map_dialog is not None:
            self._hand_map_dialog.set_selected_sensor(name)
        if scroll:
            self._scroll_sensor_into_view(name)

    def _scroll_sensor_into_view(self, name: str) -> None:
        try:
            row = self._rows[name]
            if getattr(row, "_table", None) is not None and row._table_row >= 0:
                item = row._table.item(row._table_row, 0)
                if item is not None:
                    row._table.scrollToItem(item, QtWidgets.QAbstractItemView.PositionAtCenter)
        except Exception:
            pass

    def _on_sensor_row_hovered(self, name) -> None:
        previous = getattr(self, "_table_hover_name", None)
        if previous in self._rows:
            self._rows[previous].set_table_hovered(False)
        self._table_hover_name = str(name) if name else None
        if self._table_hover_name in self._rows:
            self._rows[self._table_hover_name].set_table_hovered(True)
        if hasattr(self, "hand_map_small"):
            self.hand_map_small.set_hovered_sensor(name)
        if self._hand_map_dialog is not None:
            self._hand_map_dialog.set_hovered_sensor(name)

    def _on_svg_sensor_hovered(self, name) -> None:
        previous = self._svg_hover_name
        if previous in self._rows:
            self._rows[previous].set_svg_hovered(False)
        self._svg_hover_name = str(name) if name else None
        if self._svg_hover_name in self._rows:
            self._rows[self._svg_hover_name].set_svg_hovered(True)

    def _set_hand_map_xray(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self._settings.setValue("handmap/xray", enabled)
        if hasattr(self, "hand_xray_box") and self.hand_xray_box.isChecked() != enabled:
            self.hand_xray_box.blockSignals(True)
            self.hand_xray_box.setChecked(enabled)
            self.hand_xray_box.blockSignals(False)
        if hasattr(self, "hand_map_small"):
            self.hand_map_small.set_xray(enabled)
        if self._hand_map_dialog is not None:
            self._hand_map_dialog.set_xray(enabled)

    def _open_hand_map_dialog(self) -> None:
        # The same button (or double-click on the tiny map) toggles the large view.
        if self._hand_map_dialog is not None and self._hand_map_dialog.isVisible():
            self._hand_map_dialog.close()
            return
        if self._hand_map_dialog is None:
            mirror = self.ns.strip("/").lower() == "right"
            self._hand_map_dialog = HandMapDialog(
                self._widget, mirror=mirror, xray=bool(self.hand_xray_box.isChecked())
            )
            self._hand_map_dialog.sensorClicked.connect(lambda name: self._select_sensor(name, scroll=True))
            self._hand_map_dialog.sensorHovered.connect(self._on_svg_sensor_hovered)
            self._hand_map_dialog.xrayChanged.connect(self._set_hand_map_xray)
            self._hand_map_dialog.finished.connect(self._on_hand_map_dialog_closed)
        self._hand_map_dialog.set_selected_sensor(self._selected_name)
        self._hand_map_dialog.show()
        self._hand_map_dialog.raise_()
        self._hand_map_dialog.activateWindow()
        self.hand_map_open.setText("↙")
        self.hand_map_open.setToolTip("Close large hand/sensor map")

    def _on_hand_map_dialog_closed(self, unused=None) -> None:
        self._hand_map_dialog = None
        if hasattr(self, "hand_map_open"):
            self.hand_map_open.setText("↗")
            self.hand_map_open.setToolTip("Open large hand/sensor map")

    def _move_selection(self, delta: int) -> None:
        names = [name for name in self._channel_map if name in self._rows]
        if not names:
            return
        if self._selected_name not in names:
            index = 0
        else:
            index = names.index(self._selected_name)
        index = max(0, min(len(names) - 1, index + int(delta)))
        self._select_sensor(names[index], scroll=True)
        self._rows[names[index]].setFocus(QtCore.Qt.ShortcutFocusReason)

    def _toggle_selected_auto(self) -> None:
        if self._selected_name and self._selected_name in self._rows:
            box = self._rows[self._selected_name].auto_box
            if box.isEnabled():
                box.setChecked(not box.isChecked())

    def _set_calibration_running(self, running: bool) -> None:
        running = bool(running)
        self._calibration_running = running
        # Update the visible action first: the button must always say what the
        # next press will do, even if a later tracker reset were to fail.
        self.run_button.setText("Freeze" if running else "Start Calib")
        self.run_button.setToolTip("Freeze calibration (Space)" if running else "Start free-motion calibration (Space)")
        self.run_button.setProperty("rbmCalibRunning", running)
        self.run_button.setStyleSheet("QPushButton:checked { font-weight: bold; background: palette(highlight); color: palette(highlighted-text); }")
        if running:
            for state in self._states.values():
                state.reset_auto_trackers()
        self._refresh_hierarchy_ui()
        self._refresh_status()

    def _toggle_calibration_running(self) -> None:
        self.run_button.setChecked(not self.run_button.isChecked())

    def _set_all_sensors(self, enabled: bool) -> None:
        for row in self._rows.values():
            if row.use_box.isEnabled():
                row.use_box.setChecked(bool(enabled))
        self._refresh_hierarchy_ui()

    def _set_all_auto(self, enabled: bool) -> None:
        for row in self._rows.values():
            if row.state.supports_delta1_editing:
                row.auto_box.setChecked(bool(enabled))
        self._refresh_hierarchy_ui()

    def _refresh_hierarchy_ui(self, unused=None) -> None:
        sensor_master = bool(getattr(self, "bend_enable", None) and self.bend_enable.isChecked())
        auto_master = bool(getattr(self, "global_auto", None) and self.global_auto.isChecked())
        for row in self._rows.values():
            row.set_master_states(sensor_master, auto_master)
        if hasattr(self, "sensor_all"):
            self.sensor_all.setEnabled(sensor_master); self.sensor_none.setEnabled(sensor_master)
            self.auto_all_button.setEnabled(auto_master and sensor_master); self.auto_none_button.setEnabled(auto_master and sensor_master)
            self.collapse_button.setEnabled(sensor_master and auto_master)
            self.narrow_enabled.setEnabled(sensor_master and auto_master)
            self.narrow_rate.setEnabled(sensor_master and auto_master and self.narrow_enabled.isChecked())

    def _collapse_auto_rows(self) -> None:
        if not (self.bend_enable.isChecked() and self.global_auto.isChecked()):
            return
        changed = 0
        with self._lock:
            live_raw = dict(self._raw_values)
        for name, row in self._rows.items():
            state = self._states[name]
            if name in live_raw:
                state.raw_current = live_raw[name]
            if state.enabled and state.auto_enabled and state.collapse_to_current(seed_span_adc=2.0):
                row.sync_editors_from_state(); changed += 1
        if changed:
            self._dirty = True; self._session_modified = True
            self._apply_live_calibration()
            self._last_apply_message = "collapsed {} Auto sensor(s)".format(changed)

    def _relayout_window(self, compact: bool = False, restore_expert: bool = False) -> None:
        try:
            layout = self._widget.layout()
            if layout is not None:
                layout.invalidate(); layout.activate()
            if hasattr(self, "calibration_table"):
                self.calibration_table.doItemsLayout()
                self.calibration_table.updateGeometry()
                self.calibration_table.viewport().update()
                self._fit_calibration_column_to_viewport()
            self._widget.updateGeometry()
            hint = self._widget.sizeHint()
            current = self._widget.size()
            if restore_expert and self._expert_window_size is not None:
                target_w = max(current.width(), self._expert_window_size.width(), hint.width())
                target_h = max(self._expert_window_size.height(), hint.height())
                self._widget.resize(target_w, target_h)
            elif compact:
                target_w = max(self._widget.minimumWidth(), min(current.width(), max(hint.width(), self._widget.minimumWidth())))
                self._widget.resize(target_w, max(self._widget.minimumHeight(), hint.height()))
            elif current.height() < hint.height():
                self._widget.resize(max(current.width(), hint.width()), hint.height())

            # In standalone rqt the QWidget lives inside a dock/main-window
            # hierarchy.  One explicit top-level reflow after show/hide keeps
            # the outer window in sync without any continuous resizing.
            if self._standalone_rqt:
                top = self._widget.window()
                if top is not None and top is not self._widget:
                    top_layout = top.layout()
                    if top_layout is not None:
                        top_layout.invalidate(); top_layout.activate()
                    top.updateGeometry()
                    if compact:
                        top.adjustSize()
        except RuntimeError:
            return
        except Exception:
            pass

    def _remember_content_splitter_sizes(self, unused_pos=None, unused_index=None) -> None:
        splitter = getattr(self, "content_splitter", None)
        if splitter is None:
            return
        try:
            sizes = splitter.sizes()
            if len(sizes) >= 2 and all(int(x) >= 0 for x in sizes[:2]):
                self._settings.setValue("ui/content_splitter_sizes", "{},{}".format(int(sizes[0]), int(sizes[1])))
        except Exception:
            pass

    def _restore_content_splitter_sizes(self) -> None:
        splitter = getattr(self, "content_splitter", None)
        if splitter is None:
            return
        try:
            raw = str(self._settings.value("ui/content_splitter_sizes", ""))
            parts = [int(x.strip()) for x in raw.split(",") if x.strip()]
            if len(parts) >= 2 and sum(parts[:2]) > 0:
                splitter.setSizes(parts[:2])
        except Exception:
            pass

    def _schedule_relayout(self, compact: bool = False, restore_expert: bool = False) -> None:
        QtCore.QTimer.singleShot(0, lambda: self._relayout_window(compact=compact, restore_expert=restore_expert))
        QtCore.QTimer.singleShot(60, lambda: self._relayout_window(compact=compact, restore_expert=restore_expert))
        QtCore.QTimer.singleShot(140, lambda: self._relayout_window(compact=compact, restore_expert=restore_expert))

    def _set_expert_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if not visible and hasattr(self, "expert_container") and self.expert_container.isVisible():
            self._expert_window_size = QtCore.QSize(self._widget.size())
        if hasattr(self, "expert_container"):
            self.expert_container.setVisible(visible)
        if hasattr(self, "mapping_controls"):
            self.mapping_controls.setVisible(visible)
        if visible:
            QtCore.QTimer.singleShot(0, self._restore_content_splitter_sizes)
        self._schedule_relayout(compact=not visible, restore_expert=visible)

    def _on_table_section_resized(self, index: int, old_size: int, new_size: int) -> None:
        if self._column_resize_guard or not hasattr(self, "calibration_table"):
            return
        if self.calibration_table.isColumnHidden(index):
            return
        minimum = int(getattr(self, "_column_min_widths", {}).get(index, 24))
        if new_size < minimum:
            self._column_resize_guard = True
            try:
                self.calibration_table.setColumnWidth(index, minimum)
            finally:
                self._column_resize_guard = False
            new_size = minimum
        try:
            self._settings.setValue("table/column_{}".format(index), int(new_size))
        except Exception:
            pass
        # Rebalance the elastic calibration-bar column after any user resize.
        QtCore.QTimer.singleShot(0, self._fit_calibration_column_to_viewport)

    def _fit_calibration_column_to_viewport(self) -> None:
        """Let the bar column absorb spare viewport width, never below its user/minimum width."""
        table = getattr(self, "calibration_table", None)
        if table is None or table.isColumnHidden(2):
            return
        viewport = table.viewport()
        available = max(0, int(viewport.width()))
        if available <= 0:
            return
        other = sum(table.columnWidth(col) for col in range(table.columnCount()) if col != 2 and not table.isColumnHidden(col))
        minimum = int(self._column_min_widths.get(2, SensorRow.CALIBRATION_W))
        try:
            user_base = int(self._settings.value("table/column_2", minimum))
        except Exception:
            user_base = minimum
        target = max(minimum, user_base, available - other)
        if table.columnWidth(2) != target:
            self._column_resize_guard = True
            try:
                table.setColumnWidth(2, target)
            finally:
                self._column_resize_guard = False

    def _apply_column_visibility(self, unused=None) -> None:
        visible_map = {
            2: bool(getattr(self, "col_calibration_action", None) and self.col_calibration_action.isChecked()),
            3: bool(getattr(self, "col_current_action", None) and self.col_current_action.isChecked()),
            4: bool(getattr(self, "col_exact_action", None) and self.col_exact_action.isChecked()),
        }
        table = getattr(self, "calibration_table", None)
        if table is None:
            return
        if not hasattr(self, "_visible_column_widths"):
            self._visible_column_widths = {}
        self._column_resize_guard = True
        try:
            for col, visible in visible_map.items():
                minimum = int(self._column_min_widths.get(col, 24))
                if visible:
                    table.setColumnHidden(col, False)
                    saved = self._visible_column_widths.get(col)
                    if saved is None:
                        try:
                            saved = int(self._settings.value("table/column_{}".format(col), minimum))
                        except Exception:
                            saved = minimum
                    table.setColumnWidth(col, max(minimum, int(saved)))
                else:
                    if not table.isColumnHidden(col):
                        self._visible_column_widths[col] = max(minimum, int(table.columnWidth(col)))
                    # Hide first so the resize callback cannot re-impose the
                    # semantic minimum, then force width to zero as a Qt5/rqt
                    # compatibility measure for cell-widget columns.
                    table.setColumnHidden(col, True)
                    table.setColumnWidth(col, 0)
            table.doItemsLayout()
            table.updateGeometry()
            table.viewport().update()
        finally:
            self._column_resize_guard = False
        self._fit_calibration_column_to_viewport()
        self._schedule_relayout(compact=not bool(getattr(self, "expert_toggle", None) and self.expert_toggle.isChecked()))

    def _toggle_instructions(self, visible: bool) -> None:
        self.instructions_panel.setVisible(bool(visible))
        compact = not bool(getattr(self, "expert_toggle", None) and self.expert_toggle.isChecked()) and not bool(visible) and not bool(getattr(self, "debug_toggle", None) and self.debug_toggle.isChecked())
        self._schedule_relayout(compact=compact)

    def _reset_sensor_to_reference(self, name: str) -> None:
        """Restore one sensor from the current session reference and apply it live."""
        if name not in self._states:
            return
        points = self._session_reference_calibration.get(name)
        if not points:
            points = self._working_base_calibration.get(name)
        if not points or len(points) != 2:
            self._last_apply_message = "row reset unavailable for {}".format(name)
            self._refresh_status()
            return
        state = self._states[name]
        try:
            restored = SensorCalibrationState.from_points(
                name, points, enabled=state.enabled, auto_enabled=state.auto_enabled
            )
            state.set_points(restored.raw_a, restored.angle_a, restored.raw_b, restored.angle_b)
            state.reset_auto_trackers()
            row = self._rows.get(name)
            if row is not None:
                row.baseline_points = copy.deepcopy(list(points))
                row.sync_editors_from_state()
            self._dirty = True
            self._session_modified = (self._working_calibration() != self._baseline_calibration)
            self._apply_live_calibration()
            if self._pending_calibration_apply:
                self._last_apply_message = "{} restored; live apply pending".format(name)
            else:
                self._last_apply_message = "{} restored to session reference".format(name)
            self._refresh_status()
        except Exception as exc:
            self._last_apply_message = "row reset failed: {}".format(exc)
            rospy.logerr("Delta-1.12 row reset failed for %s: %s", name, exc)
            self._refresh_status()

    # ----- save / reset safety -------------------------------------------
    def _reset_session(self) -> None:
        self.run_button.setChecked(False)
        self._working_base_calibration = clone_calibration(self._session_reference_calibration)
        names = list(self._inventory)
        self._states = OrderedDict()
        self._rebuild_rows(names)
        self._apply_calibration_map(self._working_base_calibration, reason="session reset")
        self._dirty = False
        self._session_modified = (self._working_base_calibration != self._baseline_calibration)
        self._last_apply_monotonic = time.monotonic()

    def _load_session_calibration(self) -> None:
        start = self.calibration_file or os.path.expanduser("~/bend_calib.cal")
        filters = "Calibration / backup XML (*.cal *.cal.bak *.cal.bak-* *.bak *.xml);;Calibration (*.cal);;All files (*)"
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._widget, "Load bend calibration", start, filters
        )
        if not path:
            return
        try:
            path = os.path.abspath(path)
            calibration = parse_calibration_xml(path)
            errors = validate_calibration_map(calibration)
            if errors:
                raise CalibrationError("; ".join(errors))

            # The loaded file becomes the active session source.  This is more
            # intuitive than keeping the previously opened filename in the
            # Active field.  Backups can be loaded for replay/recovery but are
            # never overwritten directly by Save active.
            self.calibration_file = path
            self._loaded_session_source = path
            self._baseline_calibration = clone_calibration(calibration)
            self._working_base_calibration = clone_calibration(calibration)
            self._session_reference_calibration = clone_calibration(calibration)
            self._session_start_calibration = clone_calibration(calibration)
            self._calibration_order = list(calibration.keys())
            self._states = OrderedDict()
            self._rebuild_rows(list(self._inventory))
            applied = self._apply_calibration_map(self._working_calibration(), reason="loaded")
            if not applied and not self._pending_calibration_apply:
                raise CalibrationError(self._last_apply_message)
            self._dirty = False
            self._session_modified = False
            self._last_apply_monotonic = time.monotonic()
            self._refresh_header()
            self._refresh_status()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self._widget, "Load failed", str(exc))

    def _save_active(self) -> None:
        target = self.calibration_file
        if not target:
            QtWidgets.QMessageBox.warning(
                self._widget,
                "Active calibration path unknown",
                "The running calibration node did not expose path_to_calibration. "
                "Delta-1 will not guess an active file path. Use Export… for a copy.",
            )
            return

        if is_calibration_backup_path(target):
            suggested = recovery_path_for_backup(target)
            target, _ = QtWidgets.QFileDialog.getSaveFileName(
                self._widget,
                "Save recovered calibration (backup is read-only)",
                suggested,
                "Calibration XML (*.cal);;All files (*)",
            )
            if not target:
                return
            target = os.path.abspath(target)

        calibration = self._working_calibration()
        metadata = {
            "tool": "RBM Posture Calibrator Delta-1.12",
            "version": PLUGIN_VERSION,
            "namespace": self.ns,
            "serial_number": self._serial_number,
            "product_number": self._product_number,
            "calibration_param": self.calibration_param,
            "reload_service": self.reload_service_name,
        }
        try:
            result = safe_write_calibration(
                target,
                calibration,
                order=self._calibration_order,
                metadata=metadata,
            )
            # Ensure running node and disk contain the same calibration.
            if not self._apply_calibration_map(calibration, reason="saved"):
                raise CalibrationError(
                    "file was saved safely, but live reload did not succeed; the backup remains available"
                )
            self._baseline_calibration = clone_calibration(calibration)
            self._working_base_calibration = clone_calibration(calibration)
            self._session_reference_calibration = clone_calibration(calibration)
            self._session_start_calibration = clone_calibration(calibration)
            self.calibration_file = os.path.abspath(target)
            self._loaded_session_source = ""
            self._dirty = False
            self._session_modified = False
            self._last_apply_monotonic = time.monotonic()
            backup_text = result.get("backup") or "(no previous file existed)"
            QtWidgets.QMessageBox.information(
                self._widget,
                "Calibration saved safely",
                "Active file:\n{}\n\nBackup:\n{}\n\nAudit log:\n{}".format(
                    result.get("target"), backup_text, result.get("history") or "not available"
                ),
            )
            self._rebuild_rows(list(self._inventory))
        except Exception as exc:
            rospy.logerr("Delta-1 save failed: %s", exc)
            QtWidgets.QMessageBox.critical(
                self._widget,
                "Calibration save failed",
                str(exc),
            )

    def _export_copy(self, active_fallback: bool = False) -> None:
        start = self.calibration_file or os.path.expanduser("~/bend_calib.cal")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._widget,
            "Export calibration",
            start,
            "Calibration XML (*.cal);;All files (*)",
        )
        if not path:
            return
        calibration = self._working_calibration()
        try:
            result = safe_write_calibration(
                path,
                calibration,
                order=self._calibration_order,
                metadata={
                    "tool": "RBM Posture Calibrator Delta-1.12",
                    "version": PLUGIN_VERSION,
                    "namespace": self.ns,
                    "action": "export",
                },
            )
            QtWidgets.QMessageBox.information(
                self._widget,
                "Calibration exported",
                "Written:\n{}\n\nBackup (if any):\n{}".format(
                    result.get("target"), result.get("backup") or "—"
                ),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self._widget, "Export failed", str(exc))

    # ----- header / debug -------------------------------------------------
    def _refresh_header(self) -> None:
        side = self.ns.strip("/") or "global"
        count_received = len(self._inventory)
        count_enabled = sum(1 for st in self._states.values() if st.enabled)
        tactile = "—" if self._tactile_count is None else str(self._tactile_count)
        imu = "—" if self._imu_present is None else ("yes" if self._imu_present else "no")
        text = "{} · SN {} · PN {} · Bend {}/{} · Tactile {} · IMU {}".format(side.upper(), self._serial_number, self._product_number, count_received, count_enabled, tactile, imu)
        self.header_details.setFullText(text)
        if hasattr(self, "path_label"):
            self.path_label.setFullText(self.calibration_file or "active calibration path unknown")
        try:
            title = "RBM Posture Calibration — {}".format(side.upper())
            if self._serial_number not in ("", "—"):
                title += " / " + str(self._serial_number)
            self._widget.setWindowTitle(title)
            if self._standalone_rqt:
                top_window = self._widget.window()
                top_window.setWindowTitle(title)
                try:
                    posture_icon = QtGui.QIcon(resource_path(False))
                    top_window.setWindowIcon(posture_icon)
                    app = QtWidgets.QApplication.instance()
                    if app is not None:
                        app.setWindowIcon(posture_icon)
                except Exception:
                    pass
        except Exception:
            pass
        self._refresh_debug()

    def _refresh_status(self) -> None:
        dirty = "calib UNSAVED" if self._session_modified else "calib saved/baseline"
        mapping = "mapping UNSAVED" if self._mapping_dirty else "mapping saved/baseline"
        run = "CALIBRATING" if self._calibration_running else "frozen"
        loaded = (" · loaded " + os.path.basename(self._loaded_session_source)) if self._loaded_session_source else ""
        text = "{} · {} · {} · {}{}".format(run, dirty, mapping, self._last_apply_message, loaded)
        self.status_label.setFullText(text)
        palette = self.status_label.palette()
        lower = text.lower()
        if any(word in lower for word in ("failed", "blocked", "rejected", "error")):
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(190, 60, 45))
        elif "pending" in lower or "unsaved" in lower:
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(185, 120, 20))
        elif self._calibration_running:
            palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(40, 105, 165))
        else:
            palette.setColor(QtGui.QPalette.WindowText, self._widget.palette().color(QtGui.QPalette.Mid))
        self.status_label.setPalette(palette)

    def _refresh_debug(self) -> None:
        if not hasattr(self, "debug_panel"):
            return
        lines = [
            "Delta-1.12 version: {}".format(PLUGIN_VERSION),
            "namespace:       {}".format(self.ns),
            "source raw:      {}".format(self.source_raw_topic),
            "canonical raw:   {}".format(self.raw_topic),
            "cal topic:       {}".format(self.calibrated_topic),
            "channel map:     {}".format(self.channel_mapping_file or "NOT CREATED"),
            "map reload srv:  {}".format(self.mapping_reload_service_name),
            "map status:      {}".format(self._mapping_last_message),
            "calib param:     {}".format(self.calibration_param),
            "reload service:  {}".format(self.reload_service_name or "NOT FOUND"),
            "active file:     {}".format(self.calibration_file or "NOT DISCOVERED"),
            "session source:  {}".format(self._loaded_session_source or "active baseline"),
            "calib state:     {}".format("running" if self._calibration_running else "frozen"),
            "live inventory:  {}".format(", ".join(self._inventory) or "none"),
            "selected:        {}".format(self._selected_name or "none"),
            "last apply:      {}".format(self._last_apply_message),
        ]
        new_text = "\n".join(lines)
        if self.debug_panel.toPlainText() == new_text:
            return
        scroll = self.debug_panel.verticalScrollBar()
        previous_scroll = scroll.value()
        cursor = self.debug_panel.textCursor()
        previous_position = cursor.position()
        self.debug_panel.setPlainText(new_text)
        cursor = self.debug_panel.textCursor()
        cursor.setPosition(min(previous_position, len(new_text)))
        self.debug_panel.setTextCursor(cursor)
        scroll.setValue(min(previous_scroll, scroll.maximum()))

    def _toggle_debug_panel(self, visible: bool) -> None:
        self.debug_panel.setVisible(bool(visible))
        self.debug_toggle.setText("Debug / architecture")
        if visible:
            self._refresh_debug()
        compact = not bool(getattr(self, "expert_toggle", None) and self.expert_toggle.isChecked()) and not bool(visible) and not bool(getattr(self, "instructions_toggle", None) and self.instructions_toggle.isChecked())
        self._schedule_relayout(compact=compact)

    def _toggle_debug_shortcut(self) -> None:
        self.debug_toggle.setChecked(not self.debug_toggle.isChecked())

    def eventFilter(self, obj, event):  # noqa: N802
        if obj is getattr(self, "_widget", None) and event.type() == QtCore.QEvent.Close and self._standalone_rqt:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                QtCore.QTimer.singleShot(0, app.quit)
        if obj is getattr(self, "_calibration_viewport", None):
            if event.type() == QtCore.QEvent.Leave:
                self._on_sensor_row_hovered(None)
            elif event.type() == QtCore.QEvent.Resize:
                QtCore.QTimer.singleShot(0, self._fit_calibration_column_to_viewport)
        return super(SrGuiCybergloveCalibrator, self).eventFilter(obj, event)

    # ----- rqt lifecycle --------------------------------------------------
    def shutdown_plugin(self):
        """Restore last saved/session baseline if unsaved live changes remain."""
        self._shutdown = True
        try:
            if hasattr(self, "_timer"):
                self._timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "hand_xray_box"):
                self._settings.setValue("handmap/xray", bool(self.hand_xray_box.isChecked()))
        except Exception:
            pass
        try:
            if self._session_modified and self._baseline_calibration:
                self._apply_calibration_map(self._baseline_calibration, reason="shutdown restore")
        except Exception as exc:
            rospy.logwarn("Delta-1.12: could not restore calibration baseline on shutdown: %s", exc)
        try:
            if self._mapping_dirty and self._mapping_available:
                self._apply_channel_mapping(self._baseline_channel_map, reason="shutdown restore")
                self._channel_map = list(self._baseline_channel_map)
                self._mapping_dirty = False
        except Exception as exc:
            rospy.logwarn("Delta-1.12: could not restore channel mapping baseline on shutdown: %s", exc)
        for sub_name in ("_source_raw_sub", "_raw_sub", "_cal_sub", "_tactile_sub"):
            sub = getattr(self, sub_name, None)
            try:
                if sub is not None:
                    sub.unregister()
            except Exception:
                pass
        if self._standalone_rqt:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                QtCore.QTimer.singleShot(0, app.quit)

    def save_settings(self, plugin_settings, instance_settings):
        instance_settings.set_value("persistence_ms", int(self.persistence_ms.value()))
        instance_settings.set_value("narrow_enabled", bool(self.narrow_enabled.isChecked()))
        instance_settings.set_value("narrow_rate", float(self.narrow_rate.value()))
        instance_settings.set_value("apply_hz", float(self.apply_hz.value()))
        instance_settings.set_value("expert", bool(self.expert_toggle.isChecked()))
        instance_settings.set_value("hand_xray", bool(self.hand_xray_box.isChecked()))
        instance_settings.set_value("column_calibration", bool(self.col_calibration_action.isChecked()))
        instance_settings.set_value("column_current", bool(self.col_current_action.isChecked()))
        instance_settings.set_value("column_exact", bool(self.col_exact_action.isChecked()))
        try:
            instance_settings.set_value("content_splitter_sizes", ",".join(str(int(x)) for x in self.content_splitter.sizes()[:2]))
        except Exception:
            pass

    def restore_settings(self, plugin_settings, instance_settings):
        try:
            self.persistence_ms.setValue(int(instance_settings.value("persistence_ms", DEFAULT_PERSISTENCE_MS)))
            narrow = str(instance_settings.value("narrow_enabled", "true")).lower() in ("1", "true")
            self.narrow_enabled.setChecked(narrow)
            self.narrow_rate.setValue(float(instance_settings.value("narrow_rate", 30.0)))
            self.apply_hz.setValue(float(instance_settings.value("apply_hz", DEFAULT_APPLY_HZ)))
            expert = str(instance_settings.value("expert", "true")).lower() in ("1", "true")
            self.expert_toggle.setChecked(expert)
            xray = str(instance_settings.value("hand_xray", "false")).lower() in ("1", "true")
            self.hand_xray_box.setChecked(xray)
            self.col_calibration_action.setChecked(str(instance_settings.value("column_calibration", "true")).lower() in ("1", "true"))
            self.col_current_action.setChecked(str(instance_settings.value("column_current", "true")).lower() in ("1", "true"))
            self.col_exact_action.setChecked(str(instance_settings.value("column_exact", "true")).lower() in ("1", "true"))
            splitter_sizes = str(instance_settings.value("content_splitter_sizes", ""))
            if splitter_sizes:
                self._settings.setValue("ui/content_splitter_sizes", splitter_sizes)
        except Exception:
            pass
        self.run_button.setChecked(False)
        self._set_expert_visible(self.expert_toggle.isChecked())
        self._refresh_hierarchy_ui()
