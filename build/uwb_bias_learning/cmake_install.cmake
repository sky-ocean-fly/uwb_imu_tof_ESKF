# Install script for directory: /home/yu/catkin_workspace/src/uwb_bias_learning

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/yu/catkin_workspace/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/safe_execute_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/uwb_bias_learning.pc")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uwb_bias_learning/cmake" TYPE FILE FILES
    "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/uwb_bias_learningConfig.cmake"
    "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/uwb_bias_learningConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uwb_bias_learning" TYPE FILE FILES "/home/yu/catkin_workspace/src/uwb_bias_learning/package.xml")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/inspect_ros_interfaces.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/generate_pose_plan.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/list_pose_status.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/capture_static_pose.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/bag_to_raw_dataset.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/validate_raw_dataset.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/build_pose_dataset.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/make_splits.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/train_baselines.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/train_mlp.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/evaluate.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/export_model.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/predict_bias.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/estimate_range_bias_from_mocap.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/analyze_multitag_uwb_bias_bag.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/build/uwb_bias_learning/catkin_generated/installspace/plot_multitag_uwb_bias_time.py")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/uwb_bias_learning" TYPE PROGRAM FILES "/home/yu/catkin_workspace/src/uwb_bias_learning/scripts/record_session.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/uwb_bias_learning" TYPE DIRECTORY FILES
    "/home/yu/catkin_workspace/src/uwb_bias_learning/configs"
    "/home/yu/catkin_workspace/src/uwb_bias_learning/docs"
    )
endif()

