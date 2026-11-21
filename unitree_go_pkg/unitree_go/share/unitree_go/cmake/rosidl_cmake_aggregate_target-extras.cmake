# generated from rosidl_cmake/cmake/rosidl_cmake_aggregate_target-extras.cmake.in

# Create a convenience aggregate target unitree_go::unitree_go
# that links all generated interface targets, so downstream packages can use
# a single modern CMake target name instead of ${unitree_go_TARGETS}.
if(unitree_go_TARGETS AND NOT TARGET unitree_go::unitree_go)
  add_library(unitree_go::unitree_go INTERFACE IMPORTED)
  set_target_properties(unitree_go::unitree_go PROPERTIES
    INTERFACE_LINK_LIBRARIES "${unitree_go_TARGETS}")
endif()
