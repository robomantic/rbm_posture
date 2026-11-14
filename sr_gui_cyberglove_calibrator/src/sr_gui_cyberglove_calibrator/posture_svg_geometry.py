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

"""Pure SVG geometry helpers for the RBM posture sensor map.

No Qt dependency lives here.  Keeping XML/geometry generation separate makes
the alignment and hit-mask logic testable off-line and reusable by future GUIs.
"""
from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

SVG_NS = "http://www.w3.org/2000/svg"
INK_LABEL = "{http://www.inkscape.org/namespaces/inkscape}label"

# Current software identity -> anatomical element label in the supplied SVG.
# The drawing stays anatomical; historical G_* names remain a software concern.
SENSOR_TO_SVG_LABEL: Dict[str, str] = {
    "G_ThumbMPJ": "1_MCP_flex",
    "G_ThumbIJ": "1_IP",
    "G_ThumbRotate": "1_TM",
    "G_ThumbAb": "1_CMC_abd",
    "G_IndexMPJ": "2_MCP_flex",
    "G_IndexPIJ": "2_PIP",
    "G_MiddleIndexAb": "2-3_abd",
    "G_MiddleMPJ": "3_MCP_flex",
    "G_MiddlePIJ": "3_PIP",
    "G_RingMiddleAb": "3-4_abd",
    "G_RingMPJ": "4_MCP-flex",
    "G_RingPIJ": "4_PIP",
    "G_PinkieRingAb": "4-5_abd",
    "G_PinkieMPJ": "5_MCP_flex",
    "G_PinkiePIJ": "5_PIP",
    # Current ZHN compatibility/dummy channel intentionally has no physical target.
    "G_PalmArch": "",
}


def resource_path(xray: bool = False) -> str:
    name = "hand_sensors_xray.svg" if xray else "hand_sensors.svg"
    return os.path.join(os.path.dirname(__file__), "resources", name)


def _parse_svg(path: str):
    try:
        tree = ET.parse(path)
        return tree, tree.getroot()
    except Exception:
        return None, None


def _parent_map(root) -> Dict[object, object]:
    return {child: parent for parent in root.iter() for child in parent}


def svg_label_index(path: str) -> Dict[str, str]:
    """Return Inkscape label -> SVG element id for a supplied hand SVG."""
    result: Dict[str, str] = {}
    _, root = _parse_svg(path)
    if root is None:
        return result
    for element in root.iter():
        label = element.attrib.get(INK_LABEL)
        element_id = element.attrib.get("id")
        if label and element_id:
            result[str(label)] = str(element_id)
    return result


def anatomical_labels(path: str) -> Dict[str, str]:
    """Return leaf anatomical path labels for hover diagnostics.

    Group/page labels such as ``hand_lefthanded_view`` or numbered finger
    groups span large areas and made hover feedback noisy in Delta-1.5.  Sensor
    paths are handled separately by the sensor hit mask, so this diagnostic map
    contains only non-sensor anatomical *path* elements.
    """
    result: Dict[str, str] = {}
    sensor_labels = {label for label in SENSOR_TO_SVG_LABEL.values() if label}
    _, root = _parse_svg(path)
    if root is None:
        return result
    ignored = {"hand-sillouette", "hand_sillouette", "left glove", "right glove"}
    for element in root.iter():
        if element.tag.split("}")[-1] != "path":
            continue
        label = element.attrib.get(INK_LABEL)
        element_id = element.attrib.get("id")
        if not label or not element_id:
            continue
        text = str(label).strip()
        if text in sensor_labels or text.lower() in ignored:
            continue
        result[str(element_id)] = text
    return result


def _style_dict(style: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(style or "").split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _find_by_id(root, element_id: str):
    for element in root.iter():
        if element.attrib.get("id") == element_id:
            return element
    return None


def _new_svg_root(source_root):
    attrs = {}
    for key in ("viewBox", "width", "height", "preserveAspectRatio"):
        if key in source_root.attrib:
            attrs[key] = source_root.attrib[key]
    if "viewBox" not in attrs:
        attrs["viewBox"] = "0 0 {} {}".format(
            source_root.attrib.get("width", "350"),
            source_root.attrib.get("height", "500"),
        )
    return ET.Element("{{{}}}svg".format(SVG_NS), attrs)


def _clone_with_ancestor_transforms(source_root, parent_map, target, destination_parent):
    """Clone target below only the ancestor transforms needed for exact placement.

    The Delta-1.4 overlay copied only the child path.  The supplied drawing keeps
    the sensor paths in a translated layer, so omitting the parent transform
    displaced highlights and hit boxes.  This helper keeps the complete transform
    chain while intentionally copying no other visible geometry.
    """
    ancestors = []
    parent = parent_map.get(target)
    while parent is not None and parent is not source_root:
        ancestors.append(parent)
        parent = parent_map.get(parent)
    current = destination_parent
    for ancestor in reversed(ancestors):
        attrs = {}
        for key in ("transform", "clip-path", "mask", "display", "opacity"):
            if key in ancestor.attrib:
                attrs[key] = ancestor.attrib[key]
        current = ET.SubElement(current, "{{{}}}g".format(SVG_NS), attrs)
    cloned = copy.deepcopy(target)
    current.append(cloned)
    return cloned


def _set_sensor_colour(element, colour: str, interaction: bool = False) -> None:
    style = _style_dict(element.attrib.get("style", ""))
    fill = style.get("fill", element.attrib.get("fill", "none"))
    open_curve = str(fill).lower() in ("none", "transparent", "")
    if open_curve:
        # Interaction masks get a little spline-shaped tolerance, never a box.
        width = "6.0" if interaction else "4.2"
        new_style = (
            "display:inline;fill:none;stroke:{c};stroke-width:{w};"
            "stroke-linecap:round;stroke-linejoin:round;stroke-opacity:1"
        ).format(c=colour, w=width)
    else:
        new_style = (
            "display:inline;fill:{c};fill-opacity:0.78;stroke:{c};"
            "stroke-width:2.6;stroke-linejoin:round;stroke-opacity:1"
        ).format(c=colour)
    element.attrib["style"] = new_style
    # Explicit attributes must not override the style.
    for key in ("fill", "stroke", "fill-opacity", "stroke-opacity", "stroke-width"):
        element.attrib.pop(key, None)


def _overlay_svg_bytes(source_path: str, element_id: str, colour: str) -> Optional[bytes]:
    """Create an in-memory highlight using the exact source spline/path placement."""
    _, source_root = _parse_svg(source_path)
    if source_root is None:
        return None
    target = _find_by_id(source_root, element_id)
    if target is None:
        return None
    tag = target.tag.split("}")[-1]
    if tag not in ("path", "rect", "circle", "ellipse", "polygon", "polyline", "line"):
        return None
    overlay = _new_svg_root(source_root)
    cloned = _clone_with_ancestor_transforms(source_root, _parent_map(source_root), target, overlay)
    _set_sensor_colour(cloned, colour, interaction=False)
    return ET.tostring(overlay, encoding="utf-8", xml_declaration=True)




def _label_interaction_svg_bytes(source_path: str, element_labels: Dict[str, str]):
    """Return a colour-coded path mask SVG and RGB->label lookup.

    ``element_labels`` maps SVG element id -> user-facing anatomical label.  The
    same ancestor-transform preserving clone path used for sensor interaction is
    used here, so hit testing follows the *actual* SVG geometry instead of
    transformed bounding rectangles.  This fixes the historic bone/joint hover
    offset while keeping the supplied SVG files untouched.
    """
    _, source_root = _parse_svg(source_path)
    if source_root is None:
        return None, {}
    overlay = _new_svg_root(source_root)
    parents = _parent_map(source_root)
    colour_to_label: Dict[Tuple[int, int, int], str] = {}
    for idx, (element_id, label) in enumerate(sorted(element_labels.items())):
        target = _find_by_id(source_root, element_id)
        if target is None:
            continue
        # Use a disjoint deterministic palette from the sensor-mask palette.
        r = 30 + ((idx * 97 + 17) % 200)
        g = 30 + ((idx * 131 + 53) % 200)
        b = 30 + ((idx * 173 + 91) % 200)
        colour = "#{:02x}{:02x}{:02x}".format(r, g, b)
        cloned = _clone_with_ancestor_transforms(source_root, parents, target, overlay)
        _set_sensor_colour(cloned, colour, interaction=True)
        colour_to_label[(r, g, b)] = str(label)
    return ET.tostring(overlay, encoding="utf-8", xml_declaration=True), colour_to_label

def _interaction_svg_bytes(source_path: str, sensor_ids: Dict[str, str]):
    """Return one colour-coded spline mask SVG and RGB->sensor lookup.

    A single rendered mask gives precise path-shaped mouse hit testing with no
    rectangular activation areas and avoids rendering 15 SVGs on every move.
    """
    _, source_root = _parse_svg(source_path)
    if source_root is None:
        return None, {}
    overlay = _new_svg_root(source_root)
    parents = _parent_map(source_root)
    colour_to_sensor: Dict[Tuple[int, int, int], str] = {}
    for idx, (sensor, element_id) in enumerate(sorted(sensor_ids.items())):
        target = _find_by_id(source_root, element_id)
        if target is None:
            continue
        # Deterministic well-separated opaque colours; not user-visible.
        r = 40 + ((idx * 71) % 190)
        g = 40 + ((idx * 113) % 190)
        b = 40 + ((idx * 157) % 190)
        colour = "#{:02x}{:02x}{:02x}".format(r, g, b)
        cloned = _clone_with_ancestor_transforms(source_root, parents, target, overlay)
        _set_sensor_colour(cloned, colour, interaction=True)
        colour_to_sensor[(r, g, b)] = sensor
    return ET.tostring(overlay, encoding="utf-8", xml_declaration=True), colour_to_sensor
