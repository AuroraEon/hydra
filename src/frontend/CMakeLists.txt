target_sources(
  ${PROJECT_NAME}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/graph_builder.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/graph_connector.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gvd_place_extractor.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/frontier_extractor.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/mesh_segmenter.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/place_mesh_connector.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/place_2d_segmenter.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/place_2d_split_logic.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/view_database.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/view_selector.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/grid_room_segmenter.cpp
)
add_executable(test_room_segmentation test_room_segmentation.cpp)
# 链接库：包括 PCL, Hydra前端, GLog 等
target_link_libraries(test_room_segmentation
  PRIVATE
    hydra
    ${PCL_LIBRARIES}
    glog
)
include_directories(${CMAKE_SOURCE_DIR}/src/hydra/include)

# 包含目录：确保能找到 PCL 头文件
target_include_directories(test_room_segmentation PRIVATE ${PCL_INCLUDE_DIRS})