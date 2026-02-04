/* -----------------------------------------------------------------------------
 * Copyright 2022 Massachusetts Institute of Technology.
 * All Rights Reserved
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Research was sponsored by the United States Air Force Research Laboratory and
 * the United States Air Force Artificial Intelligence Accelerator and was
 * accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views
 * and conclusions contained in this document are those of the authors and should
 * not be interpreted as representing the official policies, either expressed or
 * implied, of the United States Air Force or the U.S. Government. The U.S.
 * Government is authorized to reproduce and distribute reprints for Government
 * purposes notwithstanding any copyright notation herein.
 * -------------------------------------------------------------------------- */
#include "hydra/active_window/reconstruction_module.h"

#include <config_utilities/config.h>
#include <config_utilities/printing.h>
#include <config_utilities/validation.h>

#include <chrono>
#include <fstream>
#include <filesystem>

#include "hydra/common/global_info.h"
#include "hydra/input/input_conversion.h"
#include "hydra/places/robot_footprint_integrator.h"
#include "hydra/reconstruction/mesh_integrator.h"
#include "hydra/reconstruction/projective_integrator.h"
#include "hydra/utils/timing_utilities.h"

namespace hydra {
namespace {

static const auto registration =
    config::RegistrationWithConfig<ActiveWindowModule,
                                   ReconstructionModule,
                                   ReconstructionModule::Config,
                                   ActiveWindowModule::OutputQueue::Ptr>(
        "ReconstructionModule");

double diffInSeconds(uint64_t lhs, uint64_t rhs) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             std::chrono::nanoseconds(lhs) - std::chrono::nanoseconds(rhs))
      .count();
}

}  // namespace

using timing::ScopedTimer;

void declare_config(ReconstructionModule::Config& config) {
  using namespace config;
  name("ReconstructionModule::Config");
  base<ActiveWindowModule::Config>(config);
  field(config.full_update_separation_s, "full_update_separation_s", "s");
  field(config.max_input_queue_size, "max_input_queue_size");
  field(config.tsdf, "tsdf");
  field(config.mesh, "mesh");
  config.robot_footprint.setOptional();
  field(config.robot_footprint, "robot_footprint");
}

ReconstructionModule::ReconstructionModule(const Config& config,
                                           const OutputQueue::Ptr& queue)
    : ActiveWindowModule(config, queue),
      config(config::checkValid(config)),
      last_update_ns_(std::nullopt),
      tsdf_integrator_(std::make_unique<ProjectiveIntegrator>(config.tsdf)),
      mesh_integrator_(std::make_unique<MeshIntegrator>(config.mesh)),
      footprint_integrator_(config.robot_footprint.create()) {
  if (config.tsdf.semantic_integrator && !map_.config.with_semantics) {
    LOG(ERROR)
        << "Semantic integrator specified but map does not contain semantic layer!";
  }
}

ReconstructionModule::~ReconstructionModule() {}

std::string ReconstructionModule::printInfo() const {
  return config::toString(config) + "\n" + Sink::printSinks(sinks_);
}

bool ReconstructionModule::shouldUpdate(uint64_t timestamp_ns) const {
  if (!last_update_ns_) {
    return true;
  }

  const auto diff_s = diffInSeconds(timestamp_ns, last_update_ns_.value());
  return diff_s >= config.full_update_separation_s;
}

ActiveWindowOutput::Ptr ReconstructionModule::spinOnce(const InputPacket& msg) {
  if (!msg.sensor_input) {
    LOG(ERROR) << "[Hydra Reconstruction] received invalid sensor data in input!";
    return nullptr;
  }

  const auto timestamp_ns = msg.timestamp_ns;
  const auto world_T_body = msg.world_T_body();
  const auto do_full_update = shouldUpdate(timestamp_ns);

  VLOG(2) << "[Hydra Reconstruction] starting " << (do_full_update ? "full" : "partial")
          << " update for message @ " << timestamp_ns << " (" << input_queue_->size()
          << " message(s) left)";

  ScopedTimer timer("reconstruction/spin", timestamp_ns);
  InputData::Ptr data = conversions::parseInputPacket(msg);
  if (!data) {
    return nullptr;
  }

  {  // timing scope
    ScopedTimer timer("reconstruction/tsdf", timestamp_ns);
    tsdf_integrator_->updateMap(*data, map_);
    if (footprint_integrator_) {
      footprint_integrator_->markFreespace(world_T_body.cast<float>(), map_);
    }
  }  // timing scope

  auto& tsdf = map_.getTsdfLayer();
  if (tsdf.numBlocks() == 0 || !do_full_update) {
    return nullptr;
  }

  {  // timing scope
    ScopedTimer timer("reconstruction/mesh", timestamp_ns);
    mesh_integrator_->generateMesh(map_, true, true);
  }  // timing scope

  auto output = ActiveWindowOutput::fromInput(msg);
  output->sensor_data = data;

  // this comes before clearing the update flag as we don't archive updated blocks
  if (map_window_) {
    output->archived_mesh_indices =
        map_window_->archiveBlocks(timestamp_ns, world_T_body, map_);
    VLOG(2) << "[Hydra Reconstruction] archived "
            << output->archived_mesh_indices.size() << " @ " << timestamp_ns << " [ns]";
  }

  // ================= [修改后：直接从 InputData 提取墙壁] =================
  // 前提：InputData 必须包含 vertex_map (3D点) 和 label_image (语义图)
  if (!data->vertex_map.empty() && !data->label_image.empty()) {
      const uint32_t WALL_ID = 1;
      // const uint32_t FLOOR_ID = 2; // 假设 floor 是 2，或者 simple check: 非墙壁
      const cv::Mat& points_C = data->vertex_map; // 相机/传感器坐标系下的点
      const cv::Mat& labels = data->label_image;
      
      // 获取当前传感器到世界的变换矩阵 (T_W_C)
      // 注意：data->getSensorPose() 返回的是 World_T_Sensor
      const Eigen::Isometry3d world_T_sensor = data->getSensorPose();

      // 降采样步长 (地板点太多了，没必要全要)
      int step = 4; 
      for (int r = 0; r < points_C.rows; r += step) {
          for (int c = 0; c < points_C.cols; c += step) {
              const auto& p_c = points_C.at<cv::Vec3f>(r, c);
              if (std::isnan(p_c[0]) || p_c[2] <= 0.1) continue;

              Eigen::Vector3d p_world = world_T_sensor * Eigen::Vector3d(p_c[0], p_c[1], p_c[2]);

              // 逻辑：高度在地面附近，或者 label 是地板
              // HOV-SG 逻辑：低于天花板且高于地面的所有点
              // 这里简化：只要 Z < 1.5m 且 Z > -0.5m 就认为是潜在的地板/可行使区域
              if (p_world.z() > 3.7 && p_world.z() < 5.5) {
                  output->arch_floor_points.push_back(p_world.cast<float>());
              }
          }
      }

      // 遍历像素 (注意：input data 是 dense 的图像结构)
      for (int r = 0; r < points_C.rows; ++r) {
          for (int c = 0; c < points_C.cols; ++c) {
              
              // 1. 获取语义 ID (支持 int32 或 uint8/16，视你的语义输入格式而定)
              int label = -1;
              if (labels.type() == CV_32SC1) label = labels.at<int32_t>(r, c);
              else if (labels.type() == CV_8UC1) label = labels.at<uint8_t>(r, c);
              else if (labels.type() == CV_16UC1) label = labels.at<uint16_t>(r, c);

              // 2. 语义过滤
              if (label == WALL_ID) {
                  const auto& p_c = points_C.at<cv::Vec3f>(r, c); // cv::Vec3f (x, y, z)

                  // 检查无效点 (NaN 或 深度过近/过远)
                  if (std::isnan(p_c[0]) || std::isnan(p_c[1]) || std::isnan(p_c[2]) || p_c[2] <= 0.1) {
                      continue;
                  }

                  // 3. 坐标变换：Sensor Frame -> World Frame
                  // 必须转到世界系，才能用 Z > 2.7m 进行判断
                  Eigen::Vector3d p_sensor(p_c[0], p_c[1], p_c[2]); // 这里 float 转 double 是自动的，没问题
                  Eigen::Vector3d p_world = world_T_sensor * p_sensor; // 结果是 double

                  // 4. 几何过滤：高度切片
                  if (p_world.z() > 2.8 && p_world.z() < 3.3) {
                  // if (p_world.z() > 1.5 && p_world.z() < 4.0) {
                      // [修改点]：添加 .cast<float>() 显式转换类型
                      output->arch_wall_points.push_back(p_world.cast<float>()); 
                  }
              }
          }
      }
  } else {
      VLOG(3) << "[Hydra Reconstruction] Missing vertex_map or label_image for extraction.";
  }
  // ===================================================================

  output->setMap(map_.cloneUpdated());
  for (const auto& block : tsdf) {
    block.clearUpdated();
  }

  return output;
}

}  // namespace hydra
