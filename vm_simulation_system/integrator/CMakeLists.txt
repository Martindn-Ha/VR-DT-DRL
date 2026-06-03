cmake_minimum_required(VERSION 3.0.2)
project(integrator)

find_package(catkin REQUIRED COMPONENTS message_generation std_msgs)

add_message_files(
  FILES
  BlockPose.msg
)

add_service_files(
  FILES
  SimImageCameraService.srv
  SimDepthCameraService.srv
  SupervisorGrabService.srv
  SupervisorPositionService.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs
)
