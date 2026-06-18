cmake_minimum_required(VERSION 3.8)
project(g1_description)

find_package(ament_cmake REQUIRED)

install(DIRECTORY 
  meshes 
  launch 
  rviz 
  inspire_hand 
  g1_29dof 
  images 
  urdf
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()
