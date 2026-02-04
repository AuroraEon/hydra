#include <glog/logging.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

#include "hydra/frontend/grid_room_segmenter.h"
// #include "hydra/reconstruction/volumetric_map.h" // 不需要了

using namespace hydra;

int main(int argc, char** argv) {
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);
  
  std::string pcd_path = "your_file.ply";
  if (argc > 1) pcd_path = argv[1];

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(pcd_path, *cloud) == -1) {
    LOG(ERROR) << "Couldn't read file " << pcd_path;
    return -1;
  }

  pcl::PointXYZ min_pt, max_pt;
  pcl::getMinMax3D(*cloud, min_pt, max_pt);
  
  // 处理坐标缩放 (保留原逻辑)
  double max_span = std::max({max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z});
  double scale_factor = 1.0;
  if (max_span > 100.0) scale_factor = 0.001;

  // --- 修改: 准备数据，不再建立 Voxel Map ---
  LOG(INFO) << "Converting PCL cloud to Eigen vector...";
  std::vector<Eigen::Vector3f> raw_points;
  raw_points.reserve(cloud->size());

  for (const auto& pt : cloud->points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    raw_points.emplace_back(pt.x * scale_factor, pt.y * scale_factor, pt.z * scale_factor);
  }

  // --- 配置分割器 ---
  GridRoomSegmenter::Config seg_config;
  
  // 在这里直接控制绝对高度
  double floor_z = min_pt.z * scale_factor;
  seg_config.min_height = floor_z + 2.1; // 举例：切片高度下限
  seg_config.max_height = floor_z + 2.2; // 举例：切片高度上限
  seg_config.grid_resolution = 0.05;
  seg_config.debug_save_path = "./debug_output_pcl";
  
  std::string cmd = "mkdir -p " + seg_config.debug_save_path;
  system(cmd.c_str());

  GridRoomSegmenter segmenter(seg_config);

  LOG(INFO) << "Running Room Segmentation on Point Cloud...";
  // 调用新接口
  // cv::Mat result = segmenter.computeRoomLabels(raw_points);

  // if (result.empty()) {
  //   LOG(ERROR) << "Segmentation failed (no rooms found).";
  // } else {
  //   double min_v, max_v;
  //   cv::minMaxLoc(result, &min_v, &max_v);
  //   LOG(INFO) << "Found " << max_v << " rooms.";
  // }

  return 0;
}