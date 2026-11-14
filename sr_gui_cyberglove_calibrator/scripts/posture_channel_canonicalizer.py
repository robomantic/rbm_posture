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

"""RBM posture channel canonicaliser — Delta-1.x compatibility node.

Subscribes to the driver's raw JointState vector, assigns each physical channel
an explicit sensor identity, then republishes values in the canonical order
expected by the current 16-column glove_to_human.map.

The mapping is controlled by ROS params and can be changed/reloaded live by the
Delta-1.x GUI. A per-device YAML file is used only as startup persistence; live
reload reads the validated ROS params, which makes unsaved preview reversible.

"""

from __future__ import annotations

import os
import threading

import rospy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger, TriggerResponse

from sr_gui_cyberglove_calibrator.posture_channel_mapping import (
    LEGACY_CANONICAL_ORDER_16,
    ChannelMappingError,
    infer_complete_mapping_from_names,
    identity_mapping,
    load_mapping_file,
    reorder_vector,
    validate_mapping,
)


class PostureChannelCanonicalizer:
    def __init__(self):
        self._lock = threading.RLock()
        self.mapping_file = rospy.get_param("~mapping_file", "")
        self.input_topic = rospy.get_param("~input_topic", "raw/joint_states")
        self.output_topic = rospy.get_param("~output_topic", "raw/joint_states_canonical")

        canonical, mapping, metadata, existed = load_mapping_file(
            self.mapping_file, fallback_canonical=LEGACY_CANONICAL_ORDER_16
        )
        self.canonical_order = canonical
        self.channel_to_sensor = mapping
        self._file_existed_at_start = existed
        self._have_seen_input = False
        self._mapping_valid_local = True

        # Parameters are the live editable state. GUI writes here and invokes
        # reload_mapping; restart persistence comes from mapping_file.
        rospy.set_param("~canonical_order", list(self.canonical_order))
        rospy.set_param("~channel_to_sensor", list(self.channel_to_sensor))
        rospy.set_param("~mapping_file", self.mapping_file)
        rospy.set_param("~mapping_loaded_from_file", bool(existed))
        self._mapping_valid_local = True
        rospy.set_param("~mapping_valid", True)
        rospy.set_param("~last_error", "")

        self.pub = rospy.Publisher(self.output_topic, JointState, queue_size=5)
        self.sub = rospy.Subscriber(self.input_topic, JointState, self._callback, queue_size=5)
        self.reload_srv = rospy.Service("~reload_mapping", Trigger, self._reload_service)

        rospy.loginfo(
            "Posture channel canonicalizer ready: %s -> %s (%d canonical slots, mapping %s)",
            rospy.resolve_name(self.input_topic),
            rospy.resolve_name(self.output_topic),
            len(self.canonical_order),
            "loaded from file" if existed else "identity fallback",
        )
        if self.mapping_file and not existed:
            rospy.logwarn(
                "No channel map file exists yet at %s; using identity compatibility mapping until saved by the GUI.",
                self.mapping_file,
            )

    def _set_mapping(self, canonical, mapping):
        validate_mapping(canonical, mapping, require_complete=True)
        with self._lock:
            self.canonical_order = list(canonical)
            self.channel_to_sensor = list(mapping)
        self._mapping_valid_local = True
        rospy.set_param("~mapping_valid", True)
        rospy.set_param("~last_error", "")

    def _reload_service(self, req):
        try:
            canonical = rospy.get_param("~canonical_order")
            mapping = rospy.get_param("~channel_to_sensor")
            self._set_mapping(canonical, mapping)
            return TriggerResponse(
                success=True,
                message="loaded {} channel assignments".format(len(mapping)),
            )
        except Exception as exc:
            text = str(exc)
            self._mapping_valid_local = False
            rospy.set_param("~mapping_valid", False)
            rospy.set_param("~last_error", text)
            rospy.logerr("Posture channel mapping reload rejected: %s", text)
            return TriggerResponse(success=False, message=text)

    def _maybe_adopt_incoming_names(self, msg: JointState):
        """Only before a saved file exists, adopt exact canonical name permutations.

        This preserves historical installations where the driver already publishes
        a valid canonical-name permutation, while refusing to infer identity from
        incomplete/duplicate/ad-hoc names.
        """
        if self._file_existed_at_start or self._have_seen_input:
            return
        self._have_seen_input = True
        inferred = infer_complete_mapping_from_names(msg.name, self.canonical_order)
        if inferred is not None and inferred != self.channel_to_sensor:
            try:
                self._set_mapping(self.canonical_order, inferred)
                rospy.set_param("~channel_to_sensor", list(inferred))
                rospy.loginfo(
                    "Canonicalizer adopted incoming JointState name permutation as initial channel map."
                )
            except Exception as exc:
                rospy.logwarn("Could not adopt incoming channel names: %s", exc)

    @staticmethod
    def _reorder_optional(values, canonical, mapping):
        if not values:
            return []
        if len(values) != len(mapping):
            return []
        return reorder_vector(values, canonical, mapping)

    def _callback(self, msg: JointState):
        self._maybe_adopt_incoming_names(msg)
        with self._lock:
            canonical = list(self.canonical_order)
            mapping = list(self.channel_to_sensor)
        try:
            if len(msg.position) != len(mapping):
                raise ChannelMappingError(
                    "received {} positions but channel map has {} entries".format(
                        len(msg.position), len(mapping)
                    )
                )
            out = JointState()
            out.header = msg.header
            out.name = list(canonical)
            out.position = reorder_vector(msg.position, canonical, mapping)
            out.velocity = self._reorder_optional(msg.velocity, canonical, mapping)
            out.effort = self._reorder_optional(msg.effort, canonical, mapping)
            self.pub.publish(out)
            if not self._mapping_valid_local:
                self._mapping_valid_local = True
                rospy.set_param("~mapping_valid", True)
                rospy.set_param("~last_error", "")
        except Exception as exc:
            text = str(exc)
            if self._mapping_valid_local:
                self._mapping_valid_local = False
                rospy.set_param("~mapping_valid", False)
                rospy.set_param("~last_error", text)
            rospy.logerr_throttle(2.0, "Posture canonicalizer not publishing: %s", text)


def main():
    rospy.init_node("posture_channel_canonicalizer")
    PostureChannelCanonicalizer()
    rospy.spin()


if __name__ == "__main__":
    main()
