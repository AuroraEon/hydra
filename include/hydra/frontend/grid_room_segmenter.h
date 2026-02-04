// hydra/include/hydra/frontend/grid_room_segmenter.h
#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <vector> // 必须包含 vector
#include <Eigen/Dense>
#include "hydra/reconstruction/volumetric_map.h"

namespace hydra {

class GridRoomSegmenter {
public:
  struct Config {
    double min_height = 0.5;   
    double max_height = 1.8;   
    float grid_resolution = 0.05; 
    double seed_threshold_ratio = 0.25; 
    std::string debug_save_path = "";   

    bool auto_floor_height = true;
  } config; 

  GridRoomSegmenter(const Config& config) : config(config) {}

  // 辅助函数
  Eigen::Vector3d gridToWorld(const cv::Point2i& pixel) const;

  // [新增接口] 接收 3D 墙壁点云
  void updateWallMap(const std::vector<Eigen::Vector3f>& wall_points);

  void updateFloorMap(const std::vector<Eigen::Vector3f>& floor_points);

  // [新增接口] 执行分割并返回房间信息
  void performSegmentation(uint64_t timestamp);

  // 获取最新的房间 ID 掩码 (CV_32S)
  // 修改点 1: map_mutex_ 加上 mutable 后，这里就可以在 const 函数里加锁了
  cv::Mat getLastRoomMarkers() const { 
      std::lock_guard<std::mutex> lock(map_mutex_); 
      return last_room_markers_.clone(); 
  }
    
  // 暴露坐标转换逻辑，供外部查询
  // 修改点 2: 删除了之前重复的声明，保留这个内联定义
  cv::Point2i worldToGrid(const Eigen::Vector3d& pos) const {
      // 注意：这里需要确保 grid_size_ 和 resolution_ 已经初始化
      int u = std::round(pos.x() / resolution_) + grid_size_ / 2;
      int v = std::round(pos.y() / resolution_) + grid_size_ / 2;
      return cv::Point2i(u, v);
  }

private:
  void initMap();
  double estimateFloorHeight(const VolumetricMap& map) const;

private:
  double map_resolution_;
  float resolution_ = 0.05f; 
  float map_center_x_ = 0.0f;
  float map_center_y_ = 0.0f;
  int grid_size_ = 2000; 

  cv::Mat raw_density_grid_; 
  cv::Mat raw_floor_grid_;
  cv::Mat global_grid_; 
    
  // 修改点 3: 添加 mutable 关键字
  // mutable 允许在 const 成员函数（如 getLastRoomMarkers）中修改此变量
  mutable std::mutex map_mutex_;

  cv::Mat last_room_markers_; 

  bool save_debug_data_ = true;
};

} // namespace hydra