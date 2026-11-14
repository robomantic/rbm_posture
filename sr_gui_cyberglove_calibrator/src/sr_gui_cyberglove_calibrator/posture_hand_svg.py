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

"""Qt viewer for the RBM posture sensor-map SVG (Delta-1.8)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from python_qt_binding import QtCore, QtGui, QtWidgets
try:
    from python_qt_binding import QtSvg
except Exception:  # pragma: no cover
    QtSvg = None
try:
    Signal = QtCore.Signal
except AttributeError:  # PyQt5 fallback
    Signal = QtCore.pyqtSignal

from sr_gui_cyberglove_calibrator.posture_svg_geometry import (
    SENSOR_TO_SVG_LABEL,
    _interaction_svg_bytes,
    _label_interaction_svg_bytes,
    _overlay_svg_bytes,
    anatomical_labels,
    resource_path,
    svg_label_index,
)

class HandSvgWidget(QtWidgets.QWidget):
    """Aspect-locked hand SVG with spline/path-based interaction."""

    sensorClicked = Signal(object)
    sensorHovered = Signal(object)
    elementHovered = Signal(object)
    doubleClicked = Signal()

    SELECT_COLOUR = "#d414ff"
    HOVER_COLOUR = "#00a7d8"

    def __init__(self, parent=None, mirror: bool = False, xray: bool = False):
        super(HandSvgWidget, self).__init__(parent)
        self._mirror = bool(mirror)
        self._xray = bool(xray)
        self._selected_sensor: Optional[str] = None
        self._hovered_sensor: Optional[str] = None
        self._renderer = None
        self._interaction_renderer = None
        self._interaction_colours: Dict[Tuple[int, int, int], str] = {}
        self._anatomical_renderer = None
        self._anatomical_colours: Dict[Tuple[int, int, int], str] = {}
        self._hit_image = None
        self._anatomical_hit_image = None
        self._hit_image_size = QtCore.QSize()
        self._sensor_ids: Dict[str, str] = {}
        self._anatomical: Dict[str, str] = {}
        self._highlight_renderers: Dict[Tuple[str, str], object] = {}
        self._last_render_rect = QtCore.QRectF()
        self.setMouseTracking(True)
        self.setMinimumSize(42, 54)
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        self._load_svg()

    @property
    def available(self) -> bool:
        return self._renderer is not None

    def set_mirror(self, mirror: bool) -> None:
        mirror = bool(mirror)
        if mirror != self._mirror:
            self._mirror = mirror
            self._invalidate_hit_mask()
            self.update()

    def set_xray(self, xray: bool) -> None:
        xray = bool(xray)
        if xray != self._xray:
            self._xray = xray
            self._load_svg()

    def set_selected_sensor(self, sensor: Optional[str]) -> None:
        sensor = str(sensor) if sensor else None
        if sensor != self._selected_sensor:
            self._selected_sensor = sensor
            self.update()

    def set_hovered_sensor(self, sensor: Optional[str]) -> None:
        sensor = str(sensor) if sensor else None
        if sensor != self._hovered_sensor:
            self._hovered_sensor = sensor
            self.update()

    def sizeHint(self):  # noqa: N802
        return QtCore.QSize(72, 92)

    def _invalidate_hit_mask(self) -> None:
        self._hit_image = None
        self._anatomical_hit_image = None
        self._hit_image_size = QtCore.QSize()

    def _load_svg(self) -> None:
        self._renderer = None
        self._interaction_renderer = None
        self._interaction_colours = {}
        self._anatomical_renderer = None
        self._anatomical_colours = {}
        self._sensor_ids = {}
        self._anatomical = {}
        self._highlight_renderers = {}
        self._invalidate_hit_mask()
        path = resource_path(self._xray)
        if QtSvg is None or not os.path.isfile(path):
            self.setToolTip("Qt SVG support or hand-map SVG unavailable")
            self.update()
            return
        labels = svg_label_index(path)
        self._sensor_ids = {
            sensor: labels[label]
            for sensor, label in SENSOR_TO_SVG_LABEL.items()
            if label and label in labels
        }
        self._anatomical = anatomical_labels(path)
        try:
            self._renderer = QtSvg.QSvgRenderer(path, self)
        except TypeError:
            self._renderer = QtSvg.QSvgRenderer(path)
        mask_bytes, self._interaction_colours = _interaction_svg_bytes(path, self._sensor_ids)
        if mask_bytes:
            try:
                self._interaction_renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(mask_bytes), self)
            except TypeError:
                self._interaction_renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(mask_bytes))
        anatomical_bytes, self._anatomical_colours = _label_interaction_svg_bytes(path, self._anatomical)
        if anatomical_bytes:
            try:
                self._anatomical_renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(anatomical_bytes), self)
            except TypeError:
                self._anatomical_renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(anatomical_bytes))
        self.setToolTip(
            "Hand sensor map. Click a sensor to focus its calibration row; "
            "double-click to toggle the large view."
        )
        self.update()

    def _viewbox(self) -> QtCore.QRectF:
        if self._renderer is None:
            return QtCore.QRectF(0, 0, 1, 1)
        try:
            vb = self._renderer.viewBoxF()
        except Exception:
            vb = QtCore.QRectF(self._renderer.viewBox())
        if vb.width() <= 0 or vb.height() <= 0:
            return QtCore.QRectF(0, 0, 1, 1)
        return vb

    def _fit_rect(self) -> QtCore.QRectF:
        vb = self._viewbox()
        bounds = QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if bounds.width() <= 0 or bounds.height() <= 0:
            return bounds
        scale = min(bounds.width() / vb.width(), bounds.height() / vb.height())
        w, h = vb.width() * scale, vb.height() * scale
        return QtCore.QRectF(
            bounds.x() + (bounds.width() - w) / 2.0,
            bounds.y() + (bounds.height() - h) / 2.0,
            w,
            h,
        )

    def _widget_to_svg_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        vb = self._viewbox()
        target = self._last_render_rect if not self._last_render_rect.isNull() else self._fit_rect()
        x = float(point.x())
        if self._mirror:
            x = target.right() - (x - target.left())
        sx = vb.width() / max(1e-9, target.width())
        sy = vb.height() / max(1e-9, target.height())
        return QtCore.QPointF(
            vb.x() + (x - target.x()) * sx,
            vb.y() + (point.y() - target.y()) * sy,
        )

    def _render_with_mirror(self, renderer, painter, target: QtCore.QRectF) -> None:
        painter.save()
        if self._mirror:
            painter.translate(target.center().x() * 2.0, 0.0)
            painter.scale(-1.0, 1.0)
        renderer.render(painter, target)
        painter.restore()

    def _ensure_hit_mask(self) -> None:
        if self._interaction_renderer is None or self.width() <= 0 or self.height() <= 0:
            return
        size = self.size()
        if self._hit_image is not None and self._hit_image_size == size:
            return
        image = QtGui.QImage(size, QtGui.QImage.Format_ARGB32_Premultiplied)
        image.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(image)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        target = self._fit_rect()
        self._render_with_mirror(self._interaction_renderer, painter, target)
        painter.end()
        self._hit_image = image

        if self._anatomical_renderer is not None:
            anatomical = QtGui.QImage(size, QtGui.QImage.Format_ARGB32_Premultiplied)
            anatomical.fill(QtCore.Qt.transparent)
            ap = QtGui.QPainter(anatomical)
            ap.setRenderHint(QtGui.QPainter.Antialiasing, True)
            self._render_with_mirror(self._anatomical_renderer, ap, target)
            ap.end()
            self._anatomical_hit_image = anatomical
        else:
            self._anatomical_hit_image = None
        self._hit_image_size = QtCore.QSize(size)

    def _mask_value_at(self, image, colour_map, point: QtCore.QPointF):
        if image is None:
            return None
        x, y = int(round(point.x())), int(round(point.y()))
        if x < 0 or y < 0 or x >= image.width() or y >= image.height():
            return None
        colour = image.pixelColor(x, y)
        if colour.alpha() < 20:
            return None
        rgb = (colour.red(), colour.green(), colour.blue())
        if rgb in colour_map:
            return colour_map[rgb]
        # Anti-aliased edge: accept only a nearby generated mask colour.
        best = None
        best_dist = float("inf")
        for key, value in colour_map.items():
            dist = sum((float(a)-float(b))**2 for a, b in zip(rgb, key))
            if dist < best_dist:
                best_dist, best = dist, value
        return best if best_dist <= 2500.0 else None

    def _sensor_at(self, point: QtCore.QPointF) -> Optional[str]:
        self._ensure_hit_mask()
        return self._mask_value_at(self._hit_image, self._interaction_colours, point)

    def _anatomical_at(self, point: QtCore.QPointF) -> Optional[str]:
        self._ensure_hit_mask()
        return self._mask_value_at(self._anatomical_hit_image, self._anatomical_colours, point)

    def _highlight_renderer(self, sensor: str, colour: str):
        key = (sensor, colour)
        if key in self._highlight_renderers:
            return self._highlight_renderers[key]
        element_id = self._sensor_ids.get(sensor)
        if not element_id or QtSvg is None:
            return None
        data = _overlay_svg_bytes(resource_path(self._xray), element_id, colour)
        if not data:
            return None
        try:
            renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(data), self)
        except TypeError:
            renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(data))
        self._highlight_renderers[key] = renderer
        return renderer

    def paintEvent(self, event):  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        if self._renderer is None:
            painter.setPen(self.palette().color(QtGui.QPalette.Mid))
            painter.drawText(self.rect(), int(QtCore.Qt.AlignCenter), "hand map\nunavailable")
            return
        target = self._fit_rect()
        if target != self._last_render_rect:
            self._last_render_rect = target
            self._invalidate_hit_mask()

        # Exact original supplied SVG first.
        self._render_with_mirror(self._renderer, painter, target)

        # Existing path geometry again in highlight colour, with ancestor transforms.
        if self._selected_sensor:
            renderer = self._highlight_renderer(self._selected_sensor, self.SELECT_COLOUR)
            if renderer is not None:
                self._render_with_mirror(renderer, painter, target)
        if self._hovered_sensor and self._hovered_sensor != self._selected_sensor:
            renderer = self._highlight_renderer(self._hovered_sensor, self.HOVER_COLOUR)
            if renderer is not None:
                self._render_with_mirror(renderer, painter, target)

    def resizeEvent(self, event):  # noqa: N802
        self._invalidate_hit_mask()
        super(HandSvgWidget, self).resizeEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        point = QtCore.QPointF(event.pos())
        sensor = self._sensor_at(point)
        if sensor != self._hovered_sensor:
            self._hovered_sensor = sensor
            self.sensorHovered.emit(sensor)
            self.update()
        if sensor:
            label = sensor
        else:
            label = self._anatomical_at(point)
        self.elementHovered.emit(label)
        self.setToolTip(label or "Hand sensor map")
        super(HandSvgWidget, self).mouseMoveEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        if self._hovered_sensor is not None:
            self._hovered_sensor = None
            self.sensorHovered.emit(None)
            self.update()
        self.elementHovered.emit(None)
        super(HandSvgWidget, self).leaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            sensor = self._sensor_at(QtCore.QPointF(event.pos()))
            if sensor:
                self.sensorClicked.emit(sensor)
                event.accept()
                return
        super(HandSvgWidget, self).mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if event.button() == QtCore.Qt.LeftButton:
            self.doubleClicked.emit()
            event.accept()
            return
        super(HandSvgWidget, self).mouseDoubleClickEvent(event)


class HandMapDialog(QtWidgets.QDialog):
    """Large detachable hand map for precise inspection/selection."""

    sensorClicked = Signal(object)
    sensorHovered = Signal(object)
    xrayChanged = Signal(bool)

    def __init__(self, parent=None, mirror: bool = False, xray: bool = False):
        super(HandMapDialog, self).__init__(parent)
        self.setWindowTitle("RBM Posture Sensor Map")
        self.resize(560, 720)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(7, 7, 7, 7)
        layout.setSpacing(4)
        tools = QtWidgets.QHBoxLayout()
        self.xray_box = QtWidgets.QCheckBox("X-ray")
        self.xray_box.setChecked(bool(xray))
        tools.addWidget(self.xray_box)
        tools.addStretch(1)
        layout.addLayout(tools)
        self.view = HandSvgWidget(self, mirror=mirror, xray=xray)
        self.view.setMinimumSize(360, 500)
        layout.addWidget(self.view, 1)
        self.hover_label = QtWidgets.QLabel(" ")
        self.hover_label.setAlignment(QtCore.Qt.AlignCenter)
        palette = self.hover_label.palette()
        palette.setColor(QtGui.QPalette.WindowText, palette.color(QtGui.QPalette.Mid))
        self.hover_label.setPalette(palette)
        self.hover_label.setToolTip("SVG element currently under the pointer")
        layout.addWidget(self.hover_label)
        self.xray_box.toggled.connect(self.view.set_xray)
        self.xray_box.toggled.connect(self.xrayChanged.emit)
        self.view.sensorClicked.connect(self.sensorClicked.emit)
        self.view.sensorHovered.connect(self.sensorHovered.emit)
        self.view.elementHovered.connect(self._set_hover_label)

    def _set_hover_label(self, label) -> None:
        self.hover_label.setText(str(label) if label else " ")

    def set_selected_sensor(self, sensor: Optional[str]) -> None:
        self.view.set_selected_sensor(sensor)

    def set_hovered_sensor(self, sensor: Optional[str]) -> None:
        self.view.set_hovered_sensor(sensor)
        self._set_hover_label(sensor)

    def set_xray(self, xray: bool) -> None:
        self.xray_box.blockSignals(True)
        self.xray_box.setChecked(bool(xray))
        self.xray_box.blockSignals(False)
        self.view.set_xray(bool(xray))
