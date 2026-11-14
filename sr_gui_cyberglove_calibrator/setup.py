# Copyright 2026 Gereon Buescher
# Copyright 2026 Robomantic
# SPDX-License-Identifier: GPL-2.0-only

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml.  SVG resources are package data so the
# posture sensor map also works from a catkin install space, not only source/devel.
setup_args = generate_distutils_setup(
    packages=['sr_gui_cyberglove_calibrator'],
    package_dir={'': 'src'},
    package_data={
        'sr_gui_cyberglove_calibrator': ['resources/*.svg'],
    },
)

setup(**setup_args)
