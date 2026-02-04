#include "hydra/frontend/grid_room_segmenter.h"
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <map>

namespace hydra {

void GridRoomSegmenter::initMap() {
    // 保持使用 32F 浮点型进行累加，以便后续可能的去噪（虽然现在只用简单的 >0 判断）
    raw_density_grid_ = cv::Mat::zeros(grid_size_, grid_size_, CV_32FC1);
    global_grid_ = cv::Mat::zeros(grid_size_, grid_size_, CV_8UC1);
    raw_floor_grid_ = cv::Mat::zeros(grid_size_, grid_size_, CV_32FC1);
}

void GridRoomSegmenter::updateWallMap(const std::vector<Eigen::Vector3f>& wall_points) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (raw_density_grid_.empty()) initMap();

    for (const auto& p : wall_points) {
        int u = std::round(p.x() / resolution_) + grid_size_ / 2;
        int v = std::round(p.y() / resolution_) + grid_size_ / 2;
        
        if (u >= 0 && u < grid_size_ && v >= 0 && v < grid_size_) {
            raw_density_grid_.at<float>(v, u) += 1.0f;
        }
    }
}

void GridRoomSegmenter::updateFloorMap(const std::vector<Eigen::Vector3f>& floor_points) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (raw_floor_grid_.empty()) raw_floor_grid_ = cv::Mat::zeros(grid_size_, grid_size_, CV_32FC1);

    for (const auto& p : floor_points) {
        int u = std::round(p.x() / resolution_) + grid_size_ / 2;
        int v = std::round(p.y() / resolution_) + grid_size_ / 2;
        if (u >= 0 && u < grid_size_ && v >= 0 && v < grid_size_) {
            raw_floor_grid_.at<float>(v, u) += 1.0f;
        }
    }
}

void GridRoomSegmenter::performSegmentation(uint64_t timestamp) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    if (raw_density_grid_.empty()) return;

    LOG(INFO) << "[RoomSegmenter] Starting segmentation (No Normalization)...";

    // === 步骤 1: 墙壁提取 (修改：移除归一化，使用绝对阈值) ===
    // 逻辑：只要该栅格被击中过 (value > 0.5)，就认为是墙壁 (255)
    // 解决了"观测次数少的墙壁颜色太浅"的问题
    cv::Mat walls_float_mask;
    cv::threshold(raw_density_grid_, walls_float_mask, 0.5, 255, cv::THRESH_BINARY);
    
    cv::Mat walls_skeleton;
    walls_float_mask.convertTo(walls_skeleton, CV_8UC1);

    // 简单的形态学闭运算，连接邻近的断点
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::morphologyEx(walls_skeleton, walls_skeleton, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 1);

    cv::Mat occupancy_grid = walls_skeleton.clone(); 

    // === 步骤 1.5: 墙壁修复与"充气" ===
    cv::Mat thick_grid;
    cv::dilate(occupancy_grid, thick_grid, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    int wall_gap_size = 50; 
    cv::Mat connected_h, connected_v;
    cv::morphologyEx(thick_grid, connected_h, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(wall_gap_size, 1)));
    cv::morphologyEx(thick_grid, connected_v, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, wall_gap_size)));
    
    cv::Mat connected_grid;
    cv::bitwise_or(connected_h, connected_v, connected_grid);

    int wall_inflation_size = 20;
    cv::Mat inflate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(wall_inflation_size, wall_inflation_size));
    cv::dilate(connected_grid, connected_grid, inflate_kernel);

    // === 步骤 2: 生成外部边界 (Outside Boundary) (修改：移除归一化) ===
    cv::Mat outside_boundary;
    if (!raw_floor_grid_.empty()) {
        // 修改：不使用 normalize。直接判断是否有地板数据。
        // 只要计数 > 0.5，即视为地板存在。
        cv::Mat floor_float_mask;
        cv::threshold(raw_floor_grid_, floor_float_mask, 0.5, 255, cv::THRESH_BINARY);
        
        cv::Mat floor_mask;
        floor_float_mask.convertTo(floor_mask, CV_8UC1);

        // 对二值化后的掩码进行高斯模糊或闭运算来填补空洞
        // 注意：这里模糊的是 binary mask，目的是平滑边界，而不是处理密度
        cv::GaussianBlur(floor_mask, floor_mask, cv::Size(21, 21), 2.0);
        cv::threshold(floor_mask, floor_mask, 127, 255, cv::THRESH_BINARY); // 重新二值化

        cv::Mat kernel_lg = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(floor_mask, floor_mask, cv::MORPH_CLOSE, kernel_lg, cv::Point(-1,-1), 3);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(floor_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        outside_boundary = cv::Mat::zeros(global_grid_.size(), CV_8UC1);
        cv::drawContours(outside_boundary, contours, -1, cv::Scalar(255), cv::FILLED);
    } else {
        LOG(WARNING) << "No floor data, falling back to Wall Convex Hull.";
        std::vector<cv::Point> wall_points;
        cv::findNonZero(occupancy_grid, wall_points);
        
        outside_boundary = cv::Mat::zeros(global_grid_.size(), CV_8UC1);
        if (!wall_points.empty()) {
            std::vector<std::vector<cv::Point>> hulls(1);
            cv::convexHull(wall_points, hulls[0]);
            cv::drawContours(outside_boundary, hulls, 0, cv::Scalar(255), cv::FILLED);
        }
    }

    // === 步骤 3: 组合 Full Map (Walls + Outside) ===
    cv::Mat full_obstacles;
    cv::bitwise_not(outside_boundary, full_obstacles); 
    cv::bitwise_or(walls_skeleton, full_obstacles, full_obstacles);

    cv::morphologyEx(full_obstacles, full_obstacles, cv::MORPH_CLOSE, 
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), cv::Point(-1,-1), 2);

    // === 步骤 4: 距离变换输入 ===
    cv::Mat dist_input = cv::Mat::zeros(global_grid_.size(), CV_8UC1);
    cv::bitwise_not(connected_grid, dist_input); 
    
    cv::bitwise_and(dist_input, outside_boundary, dist_input);

    cv::Mat dist_map;
    cv::distanceTransform(dist_input, dist_map, cv::DIST_L2, 5);

    double min_room_radius = 1; 
    double min_val_threshold = min_room_radius / resolution_; 
    
    double max_v;
    cv::minMaxLoc(dist_map, nullptr, &max_v);
    if (max_v < min_val_threshold) {
        LOG(WARNING) << "Max distance (" << max_v * resolution_ << "m) less than threshold.";
        return; 
    }

    cv::Mat seeds;
    cv::threshold(dist_map, seeds, min_val_threshold, 255, cv::THRESH_BINARY);
    seeds.convertTo(seeds, CV_8U);
    cv::erode(seeds, seeds, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    // === 步骤 5: 分水岭算法 ===
    cv::Mat markers;
    int num_components = cv::connectedComponents(seeds, markers);
    LOG(INFO) << "Initial Seeds Found: " << (num_components - 1);

    if (num_components <= 1) return;

    cv::Mat watershed_img;
    cv::cvtColor(dist_input, watershed_img, cv::COLOR_GRAY2BGR);
    
    int wall_label = num_components + 1;
    cv::Mat mask_bg = (markers == 0); 
    markers.setTo(wall_label, mask_bg);
    
    for(int r=0; r<markers.rows; ++r) {
        for(int c=0; c<markers.cols; ++c) {
             if (dist_input.at<uint8_t>(r,c) == 255 && markers.at<int>(r,c) == wall_label) {
                 markers.at<int>(r,c) = 0; 
             }
        }
    }

    cv::watershed(watershed_img, markers);

    // === 步骤 6: 面积过滤 ===
    std::map<int, int> area_counts;
    for(int r=0; r<markers.rows; ++r) {
        for(int c=0; c<markers.cols; ++c) {
            int idx = markers.at<int>(r, c);
            if (idx > 0 && idx != wall_label) area_counts[idx]++;
        }
    }

    double min_area_sq_meters = 2.0;
    int min_pixel_count = min_area_sq_meters / (resolution_ * resolution_);
    
    std::vector<int> valid_labels;
    for(auto const& [label, count] : area_counts) {
        if (count >= min_pixel_count) valid_labels.push_back(label);
    }
    
    LOG(INFO) << "Final Valid Rooms after Filtering: " << valid_labels.size();

    // === 步骤 7: 调试可视化 ===
    if (save_debug_data_) {
        std::string base_path = "/home/aurora/catkin_ws/src/hydra/output/uhumans2/debug_grid/"; 
        if (!std::filesystem::exists(base_path)) std::filesystem::create_directories(base_path);

        // 1. 保存墙壁图 (不再是密度图，而是清晰的二值图)
        cv::imwrite(base_path + "debug_01_walls.png", walls_skeleton);
        
        // 2. 保存外部边界图
        cv::imwrite(base_path + "debug_01b_outside_boundary.png", outside_boundary);

        // 3. 膨胀连接后的障碍物图
        cv::imwrite(base_path + "debug_02_connected_inflated.png", connected_grid);
        
        // 4. 距离变换图
        cv::Mat dist_vis;
        cv::normalize(dist_map, dist_vis, 0, 255, cv::NORM_MINMAX);
        cv::imwrite(base_path + "debug_03_distance.png", dist_vis);

        // 5. 种子图
        cv::Mat seeds_vis;
        cv::normalize(seeds, seeds_vis, 0, 255, cv::NORM_MINMAX);
        cv::imwrite(base_path + "debug_04_seeds.png", seeds_vis);

        // 6. 最终结果着色
        cv::Mat final_vis = cv::Mat::zeros(markers.size(), CV_8UC3);
        std::map<int, cv::Vec3b> color_map;
        cv::RNG rng(12345);
        for (int label : valid_labels) {
            color_map[label] = cv::Vec3b(rng.uniform(50,255), rng.uniform(50,255), rng.uniform(50,255));
        }

        for(int r=0; r<markers.rows; ++r) {
            for(int c=0; c<markers.cols; ++c) {
                int idx = markers.at<int>(r, c);
                
                bool is_valid = false;
                for(int v : valid_labels) if(v == idx) is_valid = true;

                if (idx == -1 || idx == wall_label) {
                    final_vis.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0); 
                } else if (is_valid) {
                    final_vis.at<cv::Vec3b>(r, c) = color_map[idx];
                } else {
                     final_vis.at<cv::Vec3b>(r, c) = cv::Vec3b(50, 50, 50); 
                }
            }
        }
        
        cv::Mat edges;
        cv::Canny(occupancy_grid, edges, 100, 200);
        final_vis.setTo(cv::Vec3b(255, 255, 255), edges > 0);
        
        cv::imwrite(base_path + "debug_05_final_result.png", final_vis);

        // 保存结果供 GraphBuilder 使用
        last_room_markers_ = markers.clone();
    }
}

} // namespace hydra