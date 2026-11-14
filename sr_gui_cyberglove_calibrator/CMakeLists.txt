cmake_minimum_required(VERSION 2.8.3)
project(sr_gui_cyberglove_calibrator)
find_package(catkin REQUIRED COMPONENTS rospy rqt_gui rqt_gui_py sensor_msgs std_msgs std_srvs)

catkin_package(
    DEPENDS
    CATKIN_DEPENDS rospy rqt_gui rqt_gui_py sensor_msgs std_msgs std_srvs
    INCLUDE_DIRS
    LIBRARIES
)
catkin_python_setup()

catkin_install_python(
  PROGRAMS scripts/posture_channel_canonicalizer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install( DIRECTORY uis resource images DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )

install( FILES sr_cyberglove_calibrator_plugin.xml sr_cyberglove_tweaker_plugin.xml DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )

install( FILES THIRD_PARTY_NOTICES.md DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_cyberglove_calibrator.py)
endif()
