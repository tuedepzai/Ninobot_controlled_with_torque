cmake_minimum_required(VERSION 3.8)
project(nino_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch meshes urdf worlds
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

install(
  PROGRAMS scripts/gz_sim_clean
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
