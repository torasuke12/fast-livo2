/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "voxel_map.h"
#include "point_uncertainty.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

namespace
{
constexpr int kFastLioNumMatchPoints = 5;
constexpr double kFastLioMaxNeighborSquaredDistance = 5.0;
constexpr double kFastLioPlaneThreshold = 0.1;
constexpr double kFastLioResidualScoreThreshold = 0.9;
constexpr double kFastLioLaserPointCovariance = 0.001;
constexpr double kFastLioMoveThreshold = 1.5;

VOXEL_LOCATION worldToVoxelLocation(const Eigen::Vector3d &point, const double voxel_size)
{
  return VOXEL_LOCATION(static_cast<int64_t>(std::floor(point.x() / voxel_size)),
                        static_cast<int64_t>(std::floor(point.y() / voxel_size)),
                        static_cast<int64_t>(std::floor(point.z() / voxel_size)));
}

M3D so3RetractionJacobian(const V3D &delta)
{
  const double squared_norm = delta.squaredNorm();
  if (squared_norm < 1e-16) return M3D::Identity();

  const double norm = std::sqrt(squared_norm);
  M3D delta_hat;
  delta_hat << SKEW_SYM_MATRX(delta);
  return M3D::Identity() + (1.0 - std::cos(norm)) / squared_norm * delta_hat +
         (1.0 - std::sin(norm) / norm) / squared_norm * delta_hat * delta_hat;
}

M3D gravityTangentTransport(const V3D &from_gravity, const V3D &to_gravity)
{
  if (from_gravity.norm() < 1e-9 || to_gravity.norm() < 1e-9) return M3D::Identity();

  const V3D from_unit = from_gravity.normalized();
  const V3D to_unit = to_gravity.normalized();
  const Eigen::Quaterniond rotation = Eigen::Quaterniond::FromTwoVectors(from_unit, to_unit);
  const M3D to_tangent = M3D::Identity() - to_unit * to_unit.transpose();
  return to_tangent * rotation.toRotationMatrix();
}
} // namespace

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov)
{
  if (pb[2] == 0) pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
}

void loadVoxelConfig(const rclcpp::Node::SharedPtr &node, VoxelMapConfig &voxel_config)
{
  voxel_config.is_pub_plane_map_ = fast_livo2::declareOrGet<bool>(node, "publish/pub_plane_en", false);

  voxel_config.max_layer_ = fast_livo2::declareOrGet<int>(node, "lio/max_layer", 1);
  voxel_config.max_voxel_size_ = fast_livo2::declareOrGet<double>(node, "lio/voxel_size", 0.5);
  voxel_config.planner_threshold_ = fast_livo2::declareOrGet<double>(node, "lio/min_eigen_value", 0.01);
  voxel_config.sigma_num_ = fast_livo2::declareOrGet<double>(node, "lio/sigma_num", 3.0);
  voxel_config.max_state_cov_for_matching_ = fast_livo2::declareOrGet<double>(node, "lio/max_state_cov_for_matching", 0.01);
  if (voxel_config.max_state_cov_for_matching_ <= 0.0) voxel_config.max_state_cov_for_matching_ = 0.01;
  voxel_config.neighbor_voxel_radius_ = fast_livo2::declareOrGet<int>(node, "lio/neighbor_voxel_radius", 1);
  if (voxel_config.neighbor_voxel_radius_ < 0) voxel_config.neighbor_voxel_radius_ = 0;
  voxel_config.correspondence_mode_ =
      fast_livo2::declareOrGet<std::string>(node, "lio/correspondence_mode", "fastlio_ikdtree");
  if (voxel_config.correspondence_mode_ == "fastlio_knn")
  {
    RCLCPP_WARN(node->get_logger(),
                "lio/correspondence_mode=fastlio_knn is deprecated; using the FAST-LIO incremental ikd-tree");
    voxel_config.correspondence_mode_ = "fastlio_ikdtree";
  }
  if (voxel_config.correspondence_mode_ != "fastlio_ikdtree" && voxel_config.correspondence_mode_ != "voxel_plane")
  {
    RCLCPP_WARN(node->get_logger(), "Unknown lio/correspondence_mode '%s'; using fastlio_ikdtree",
                voxel_config.correspondence_mode_.c_str());
    voxel_config.correspondence_mode_ = "fastlio_ikdtree";
  }
  voxel_config.fastlio_filter_size_map_ =
      fast_livo2::declareOrGet<double>(node, "lio/fastlio_filter_size_map", 0.5);
  if (voxel_config.fastlio_filter_size_map_ <= 0.0) voxel_config.fastlio_filter_size_map_ = 0.5;
  voxel_config.fastlio_cube_side_length_ =
      fast_livo2::declareOrGet<double>(node, "lio/fastlio_cube_side_length", 1000.0);
  if (voxel_config.fastlio_cube_side_length_ <= 0.0) voxel_config.fastlio_cube_side_length_ = 1000.0;
  voxel_config.fastlio_detection_range_ =
      fast_livo2::declareOrGet<double>(node, "lio/fastlio_detection_range", 100.0);
  if (voxel_config.fastlio_detection_range_ <= 0.0) voxel_config.fastlio_detection_range_ = 100.0;
  voxel_config.gravity_norm_constraint_ =
      fast_livo2::declareOrGet<bool>(node, "lio/gravity_norm_constraint", true);
  voxel_config.debug_log_dir_ = fast_livo2::declareOrGet<std::string>(node, "debug/log_dir", "");
  auto &normal = voxel_config.normal_consistency_;
  normal.enabled = fast_livo2::declareOrGet<bool>(node, "lio/normal_consistency/enabled", false);
  normal.neighbors = fast_livo2::declareOrGet<int>(node, "lio/normal_consistency/neighbors", 20);
  normal.max_neighbor_distance = fast_livo2::declareOrGet<double>(node, "lio/normal_consistency/max_neighbor_distance", 1.0);
  normal.max_planarity_ratio = fast_livo2::declareOrGet<double>(node, "lio/normal_consistency/max_planarity_ratio", 0.1);
  normal.min_spread_ratio = fast_livo2::declareOrGet<double>(node, "lio/normal_consistency/min_spread_ratio", 0.1);
  normal.max_angle_deg = fast_livo2::declareOrGet<double>(node, "lio/normal_consistency/max_angle_deg", 30.0);
  normal.validate();
  if (normal.enabled && voxel_config.correspondence_mode_ != "fastlio_ikdtree")
    throw std::invalid_argument("lio.normal_consistency supports fastlio_ikdtree/fastlio_knn only");
  RCLCPP_INFO(node->get_logger(), "Source/map normal gate: %s (%.1f deg, %d source neighbors)",
              normal.enabled ? "enabled" : "disabled", normal.max_angle_deg, normal.neighbors);
  auto &xicp = voxel_config.xicp_;
  xicp.enabled = fast_livo2::declareOrGet<bool>(node, "lio/xicp/enabled", false);
  xicp.diagnostics = fast_livo2::declareOrGet<bool>(node, "lio/xicp/diagnostics", false);
  xicp.contribution_min = fast_livo2::declareOrGet<double>(node, "lio/xicp/contribution_min", xicp.contribution_min);
  xicp.contribution_strong = fast_livo2::declareOrGet<double>(node, "lio/xicp/contribution_strong", xicp.contribution_strong);
  xicp.enough_information = fast_livo2::declareOrGet<double>(node, "lio/xicp/enough_information", xicp.enough_information);
  xicp.insufficient_information = fast_livo2::declareOrGet<double>(node, "lio/xicp/insufficient_information", xicp.insufficient_information);
  xicp.minimal_information = fast_livo2::declareOrGet<double>(node, "lio/xicp/minimal_information", xicp.minimal_information);
  xicp.rank_relative_tolerance = fast_livo2::declareOrGet<double>(node, "lio/xicp/rank_relative_tolerance", xicp.rank_relative_tolerance);
  xicp.validate();
  RCLCPP_INFO(node->get_logger(), "X-ICP constrained LIO update: %s (diagnostics %s)",
              xicp.enabled ? "enabled" : "disabled", xicp.diagnostics ? "enabled" : "disabled");
  auto &diagnostics = voxel_config.diagnostics_;
  diagnostics.enabled = fast_livo2::declareOrGet<bool>(node, "lio/diagnostics/enabled", false);
  diagnostics.output_dir = fast_livo2::declareOrGet<std::string>(node, "lio/diagnostics/output_dir", "");
  diagnostics.detail_enabled = fast_livo2::declareOrGet<bool>(node, "lio/diagnostics/detail/enabled", false);
  diagnostics.detail_final_only = fast_livo2::declareOrGet<bool>(node, "lio/diagnostics/detail/final_only", false);
  diagnostics.geometry_enabled = fast_livo2::declareOrGet<bool>(node, "lio/diagnostics/geometry/enabled", false);
  diagnostics.detail_start_sec = fast_livo2::declareOrGet<double>(node, "lio/diagnostics/detail/start_sec", 170.0);
  diagnostics.detail_end_sec = fast_livo2::declareOrGet<double>(node, "lio/diagnostics/detail/end_sec", 220.0);
  diagnostics.xicp_enabled = xicp.enabled;
  diagnostics.validate();
  diagnostics.correspondence_mode = voxel_config.correspondence_mode_;
  auto &initialization = voxel_config.initialization_;
  initialization.min_points = fast_livo2::declareOrGet<int>(node, "lio/voxel_init_min_points", 5000);
  initialization.min_planes = fast_livo2::declareOrGet<int>(node, "lio/voxel_init_min_planes", 100);
  initialization.min_normal_eigenvalue =
      fast_livo2::declareOrGet<double>(node, "lio/voxel_init_min_normal_eigenvalue", 0.02);
  initialization.min_duration_sec = fast_livo2::declareOrGet<double>(node, "lio/voxel_init_min_duration_sec", 0.1);
  initialization.max_duration_sec = fast_livo2::declareOrGet<double>(node, "lio/voxel_init_max_duration_sec", 2.0);
  initialization.max_points = fast_livo2::declareOrGet<int>(node, "lio/voxel_init_max_points", 100000);
  if (voxel_config.correspondence_mode_ == "voxel_plane" &&
      (initialization.min_points < 6 || initialization.min_planes < 3 ||
      initialization.max_points < initialization.min_points ||
      !std::isfinite(initialization.min_normal_eigenvalue) || initialization.min_normal_eigenvalue <= 0.0 ||
      initialization.min_normal_eigenvalue >= 1.0 / 3.0 ||
      !std::isfinite(initialization.min_duration_sec) || initialization.min_duration_sec < 0.0 ||
      !std::isfinite(initialization.max_duration_sec) || initialization.max_duration_sec <= initialization.min_duration_sec))
  {
    throw std::invalid_argument("Invalid lio.voxel_init_* thresholds: require points >= 6, planes >= 3, "
                                "max_points >= min_points, 0 < normal eigenvalue < 1/3, "
                                "and 0 <= min_duration < max_duration");
  }
  RCLCPP_INFO(node->get_logger(), "LIO correspondence mode: %s", voxel_config.correspondence_mode_.c_str());
  voxel_config.beam_err_ = fast_livo2::declareOrGet<double>(node, "lio/beam_err", 0.02);
  voxel_config.dept_err_ = fast_livo2::declareOrGet<double>(node, "lio/dept_err", 0.05);
  const auto layer_init_num = fast_livo2::declareOrGet<std::vector<int64_t>>(node, "lio/layer_init_num", {5, 5, 5, 5, 5});
  voxel_config.layer_init_num_.clear();
  voxel_config.layer_init_num_.reserve(layer_init_num.size());
  for (const auto value : layer_init_num)
  {
    voxel_config.layer_init_num_.push_back(static_cast<int>(value));
  }
  voxel_config.max_points_num_ = fast_livo2::declareOrGet<int>(node, "lio/max_points_num", 50);
  voxel_config.max_iterations_ = fast_livo2::declareOrGet<int>(node, "lio/max_iterations", 5);

  voxel_config.map_sliding_en = fast_livo2::declareOrGet<bool>(node, "local_map/map_sliding_en", false);
  voxel_config.half_map_size = fast_livo2::declareOrGet<int>(node, "local_map/half_map_size", 100);
  voxel_config.sliding_thresh = fast_livo2::declareOrGet<double>(node, "local_map/sliding_thresh", 8.0);
  if (diagnostics.enabled)
  {
    for (const auto &name : node->list_parameters({}, 0).names)
      diagnostics.parameters.emplace(name, node->get_parameter(name).value_to_string());
    RCLCPP_INFO(node->get_logger(), "LIO diagnostics enabled (detail %s, [%.3f, %.3f) s)",
                diagnostics.detail_enabled ? "enabled" : "disabled",
                diagnostics.detail_start_sec, diagnostics.detail_end_sec);
  }
}

void VoxelOctoTree::init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane)
{
  plane->plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance_ = Eigen::Matrix3d::Zero();
  plane->center_ = Eigen::Vector3d::Zero();
  plane->normal_ = Eigen::Vector3d::Zero();
  plane->points_size_ = points.size();
  plane->radius_ = 0;
  for (auto pv : points)
  {
    plane->covariance_ += pv.point_w * pv.point_w.transpose();
    plane->center_ += pv.point_w;
  }
  plane->center_ = plane->center_ / plane->points_size_;
  plane->covariance_ = plane->covariance_ / plane->points_size_ - plane->center_ * plane->center_.transpose();
  Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance_);
  Eigen::Matrix3cd evecs = es.eigenvectors();
  Eigen::Vector3cd evals = es.eigenvalues();
  Eigen::Vector3d evalsReal;
  evalsReal = evals.real();
  Eigen::Matrix3f::Index evalsMin, evalsMax;
  evalsReal.rowwise().sum().minCoeff(&evalsMin);
  evalsReal.rowwise().sum().maxCoeff(&evalsMax);
  // Accumulating static scans can produce coincident/collinear samples. They
  // do not define a plane: avoid an invalid middle index or zero eigen-gap in
  // the normal-covariance Jacobian below.
  if (!evalsReal.allFinite() || evalsMin == evalsMax)
  {
    plane->is_plane_ = false;
    plane->is_update_ = true;
    return;
  }
  int evalsMid = 3 - evalsMin - evalsMax;
  if (evalsReal(evalsMid) - evalsReal(evalsMin) <= 1e-9)
  {
    plane->is_plane_ = false;
    plane->is_update_ = true;
    return;
  }
  Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
  Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
  Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_, 0, 0, 0, 1.0 / plane->points_size_;
  // && evalsReal(evalsMid) > 0.05
  //&& evalsReal(evalsMid) > 0.01
  if (evalsReal(evalsMin) < planer_threshold_)
  {
    for (int i = 0; i < points.size(); i++)
    {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++)
      {
        if (m != (int)evalsMin)
        {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].point_w - plane->center_).transpose() / ((plane->points_size_) * (evalsReal[evalsMin] - evalsReal[m])) *
              (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() + evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
          F.row(m) = F_m;
        }
        else
        {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs.real() * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_var_ += J * points[i].var * J.transpose();
    }

    plane->normal_ << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
    plane->y_normal_ << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
    plane->x_normal_ << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
    plane->min_eigen_value_ = evalsReal(evalsMin);
    plane->mid_eigen_value_ = evalsReal(evalsMid);
    plane->max_eigen_value_ = evalsReal(evalsMax);
    plane->radius_ = sqrt(evalsReal(evalsMax));
    plane->d_ = -(plane->normal_(0) * plane->center_(0) + plane->normal_(1) * plane->center_(1) + plane->normal_(2) * plane->center_(2));
    plane->is_plane_ = true;
    plane->is_update_ = true;
    if (!plane->is_init_)
    {
      plane->id_ = voxel_plane_id;
      voxel_plane_id++;
      plane->is_init_ = true;
    }
  }
  else
  {
    plane->is_update_ = true;
    plane->is_plane_ = false;
  }
}

void VoxelOctoTree::init_octo_tree()
{
  if (temp_points_.size() > points_size_threshold_)
  {
    init_plane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane_ == true)
    {
      octo_state_ = 0;
      // new added
      if (temp_points_.size() > max_points_num_)
      {
        update_enable_ = false;
        std::vector<pointWithVar>().swap(temp_points_);
        new_points_ = 0;
      }
    }
    else
    {
      octo_state_ = 1;
      cut_octo_tree();
    }
    init_octo_ = true;
    new_points_ = 0;
  }
}

void VoxelOctoTree::cut_octo_tree()
{
  if (layer_ >= max_layer_)
  {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
    if (temp_points_[i].point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
    if (temp_points_[i].point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_++;
  }
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->points_size_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane_)
        {
          leaves_[i]->octo_state_ = 0;
          // new added
          if (leaves_[i]->temp_points_.size() > leaves_[i]->max_points_num_)
          {
            leaves_[i]->update_enable_ = false;
            std::vector<pointWithVar>().swap(leaves_[i]->temp_points_);
            new_points_ = 0;
          }
        }
        else
        {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->init_octo_ = true;
        leaves_[i]->new_points_ = 0;
      }
    }
  }
}

void VoxelOctoTree::UpdateOctoTree(const pointWithVar &pv)
{
  if (!init_octo_)
  {
    new_points_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > points_size_threshold_) { init_octo_tree(); }
  }
  else
  {
    if (plane_ptr_->is_plane_)
    {
      if (update_enable_)
      {
        new_points_++;
        temp_points_.push_back(pv);
        if (new_points_ > update_size_threshold_)
        {
          init_plane(temp_points_, plane_ptr_);
          new_points_ = 0;
        }
        if (temp_points_.size() >= max_points_num_)
        {
          update_enable_ = false;
          std::vector<pointWithVar>().swap(temp_points_);
          new_points_ = 0;
        }
      }
    }
    else
    {
      if (layer_ < max_layer_)
      {
        int xyz[3] = {0, 0, 0};
        if (pv.point_w[0] > voxel_center_[0]) { xyz[0] = 1; }
        if (pv.point_w[1] > voxel_center_[1]) { xyz[1] = 1; }
        if (pv.point_w[2] > voxel_center_[2]) { xyz[2] = 1; }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) { leaves_[leafnum]->UpdateOctoTree(pv); }
        else
        {
          leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
          leaves_[leafnum]->layer_init_num_ = layer_init_num_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else
      {
        if (update_enable_)
        {
          new_points_++;
          temp_points_.push_back(pv);
          if (new_points_ > update_size_threshold_)
          {
            init_plane(temp_points_, plane_ptr_);
            new_points_ = 0;
          }
          if (temp_points_.size() > max_points_num_)
          {
            update_enable_ = false;
            std::vector<pointWithVar>().swap(temp_points_);
            new_points_ = 0;
          }
        }
      }
    }
  }
}

VoxelOctoTree *VoxelOctoTree::find_correspond(Eigen::Vector3d pw)
{
  if (!init_octo_ || plane_ptr_->is_plane_ || (layer_ >= max_layer_)) return this;

  int xyz[3] = {0, 0, 0};
  xyz[0] = pw[0] > voxel_center_[0] ? 1 : 0;
  xyz[1] = pw[1] > voxel_center_[1] ? 1 : 0;
  xyz[2] = pw[2] > voxel_center_[2] ? 1 : 0;
  int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

  // printf("leafnum: %d. \n", leafnum);

  return (leaves_[leafnum] != nullptr) ? leaves_[leafnum]->find_correspond(pw) : this;
}

VoxelOctoTree *VoxelOctoTree::Insert(const pointWithVar &pv)
{
  if ((!init_octo_) || (init_octo_ && plane_ptr_->is_plane_) || (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ >= max_layer_)))
  {
    new_points_++;
    temp_points_.push_back(pv);
    return this;
  }

  if (init_octo_ && (!plane_ptr_->is_plane_) && (layer_ < max_layer_))
  {
    int xyz[3] = {0, 0, 0};
    xyz[0] = pv.point_w[0] > voxel_center_[0] ? 1 : 0;
    xyz[1] = pv.point_w[1] > voxel_center_[1] ? 1 : 0;
    xyz[2] = pv.point_w[2] > voxel_center_[2] ? 1 : 0;
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr) { return leaves_[leafnum]->Insert(pv); }
    else
    {
      leaves_[leafnum] = new VoxelOctoTree(max_layer_, layer_ + 1, layer_init_num_[layer_ + 1], max_points_num_, planer_threshold_);
      leaves_[leafnum]->layer_init_num_ = layer_init_num_;
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
      return leaves_[leafnum]->Insert(pv);
    }
  }
  return nullptr;
}

void VoxelMapManager::PrepareSourceNormals(bool geometry_requested)
{
  source_normals_.clear();
  normal_stats_ = {};
  if (config_setting_.correspondence_mode_ != "fastlio_ikdtree" ||
      (!config_setting_.normal_consistency_.enabled && !geometry_requested)) return;
  const auto start = std::chrono::steady_clock::now();
  try
  {
    source_normals_ = fast_livo2::normal_consistency::estimate(*feats_down_body_, config_setting_.normal_consistency_);
  }
  catch (const std::exception &error)
  {
    if (config_setting_.normal_consistency_.enabled) throw;
    // Optional geometry diagnostics must not turn a logging failure into a filter failure.
    lio_diagnostics_.disable(error.what());
    return;
  }
  for (const auto &source : source_normals_)
    source.usable() ? ++normal_stats_.valid : ++normal_stats_.invalid;
  normal_stats_.computation_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
}

void VoxelMapManager::StateEstimation(StatesGroup &state_propagat, const double scan_time)
{
  using DiagnosticStatus = fast_livo2::diagnostics::ScanStatus;
  DiagnosticStatus diagnostic_status = DiagnosticStatus::NoUpdate;
  int diagnostic_last_iteration = -2;
  if (lio_diagnostics_.enabled()) lio_diagnostics_.beginScan(scan_time, state_propagat.cov);
  if (config_setting_.correspondence_mode_ == "fastlio_ikdtree")
  {
    if (fastlio_ikdtree_.Root_Node == nullptr)
    {
      effct_feat_num_ = 0;
      if (lio_diagnostics_.enabled()) lio_diagnostics_.endScan(DiagnosticStatus::NoMap, -2, state_.cov);
      return;
    }
    SegmentFastLioLocalMap();
  }

  cross_mat_list_.clear();
  cross_mat_list_.reserve(feats_down_size_);
  body_cov_list_.clear();
  body_cov_list_.reserve(feats_down_size_);

  // build_residual_time = 0.0;
  // ekf_time = 0.0;
  // double t0 = omp_get_wtime();

  for (size_t i = 0; i < feats_down_body_->size(); i++)
  {
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    if (point_this[2] == 0) { point_this[2] = 0.001; }
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    body_cov_list_.push_back(var);
    point_this = extR_ * point_this + extT_;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    cross_mat_list_.push_back(point_crossmat);
  }

  vector<pointWithVar>().swap(pv_list_);
  pv_list_.resize(feats_down_size_);

  int rematch_num = 0;
  MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE;
  G.setZero();
  H_T_H.setZero();
  I_STATE.setIdentity();

  bool flg_EKF_converged, EKF_stop_flg = 0;
  bool recompute_fastlio_neighbors = true;
  const MD(DIM_STATE, DIM_STATE) propagated_covariance = state_propagat.cov;
  fastlio_nearest_points_.assign(feats_down_size_, {});
  fastlio_neighbor_valid_.assign(feats_down_size_, 0);
  diagnostic_geometry_.clear();
  if (lio_diagnostics_.geometryEnabled())
  {
    try { diagnostic_geometry_.resize(feats_down_size_); }
    catch (const std::exception &error) { lio_diagnostics_.disable(error.what()); }
    catch (...) { lio_diagnostics_.disable("geometry allocation failed"); }
  }
  PrepareSourceNormals(lio_diagnostics_.geometryEnabled());
  // FAST-LIO's IKFoM update loops from -1 through max_iterations - 1.
  for (int iterCount = -1; iterCount < config_setting_.max_iterations_; iterCount++)
  {
    diagnostic_last_iteration = iterCount;
    double total_residual = 0.0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    TransformLidar(state_.rot_end, state_.pos_end, feats_down_body_, world_lidar);
    const auto pose_cov=fast_livo2::boundedPoseCovariance(state_.cov.topLeftCorner<6,6>(),
        config_setting_.max_state_cov_for_matching_);
    for (size_t i = 0; i < feats_down_body_->size(); i++)
    {
      pointWithVar &pv = pv_list_[i];
      pv.point_b << feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z;
      pv.point_w << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;

      const M3D cov=fast_livo2::worldPointCovariance(extR_*pv.point_b+extT_,state_.rot_end,extR_,
          body_cov_list_[i],pose_cov);
      pv.var = cov;
      pv.body_var = body_cov_list_[i];
      if (lio_diagnostics_.geometryEnabled())
      {
        auto &point = diagnostic_geometry_[i];
        if (config_setting_.correspondence_mode_ == "voxel_plane") point = {};
        point.lidar = pv.point_b;
        point.world = pv.point_w;
        point.match_index = -1;
        point.status = fast_livo2::diagnostics::MatchStatus::InvalidPoint;
        point.normal.setZero(); point.center.setZero();
        point.plane_d = 0.0; point.distance = 0.0;
        point.source = source_normals_.empty() ? fast_livo2::normal_consistency::SourceNormal{} : source_normals_[i];
        point.comparison = {};
      }
    }
    ptpl_list_.clear();

    // double t1 = omp_get_wtime();

    if (config_setting_.correspondence_mode_ == "fastlio_ikdtree")
    {
      BuildResidualListFastLio(pv_list_, ptpl_list_, recompute_fastlio_neighbors, iterCount);
    }
    else
    {
      BuildResidualListOMP(pv_list_, ptpl_list_);
    }

    // build_residual_time += omp_get_wtime() - t1;

    for (int i = 0; i < ptpl_list_.size(); i++)
    {
      total_residual += fabs(ptpl_list_[i].dis_to_plane_);
    }
    effct_feat_num_ = ptpl_list_.size();
    cout << "[ LIO ] Raw feature num: " << feats_undistort_->size() << ", downsampled feature num:" << feats_down_size_ 
         << " effective feature num: " << effct_feat_num_ << " average residual: "
         << (effct_feat_num_ > 0 ? total_residual / effct_feat_num_ : 0.0) << endl;
    if (effct_feat_num_ < 1)
    {
      // An empty rematch must not retain an earlier iteration's uncommitted
      // correction with the propagated covariance.
      const StatesGroup diagnostic_before = state_;
      state_ = state_propagat;
      captureGeometry(iterCount, state_propagat, diagnostic_before, state_);
      diagnostic_status = DiagnosticStatus::NoMatches;
      break;
    }

    /*** Computation of Measuremnt Jacobian matrix H and measurents covarience
     * ***/
    MatrixXd Hsub(effct_feat_num_, 6);
    MatrixXd Hsub_T_R_inv(6, effct_feat_num_);
    VectorXd R_inv(effct_feat_num_);
    VectorXd meas_vec(effct_feat_num_);
    meas_vec.setZero();
    for (int i = 0; i < effct_feat_num_; i++)
    {
      auto &ptpl = ptpl_list_[i];
      V3D point_this(ptpl.point_b_);
      point_this = extR_ * point_this + extT_;
      V3D point_body(ptpl.point_b_);
      M3D point_crossmat;
      point_crossmat << SKEW_SYM_MATRX(point_this);

      /*** get the normal vector of closest surface/corner ***/

      V3D point_world = state_propagat.rot_end * point_this + state_propagat.pos_end;
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point_world - ptpl_list_[i].center_;
      J_nq.block<1, 3>(0, 3) = -ptpl_list_[i].normal_;

      M3D var;
      // V3D normal_b = state_.rot_end.inverse() * ptpl_list_[i].normal_;
      // V3D point_b = ptpl_list_[i].point_b_;
      // double cos_theta = fabs(normal_b.dot(point_b) / point_b.norm());
      // ptpl_list_[i].body_cov_ = ptpl_list_[i].body_cov_ * (1.0 / cos_theta) * (1.0 / cos_theta);

      // point_w cov
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) + (-point_crossmat) * state_propagat.cov.block<3, 3>(0, 0) * (-point_crossmat).transpose();

      // point_w cov (another_version)
      // var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose() +
      //       state_propagat.cov.block<3, 3>(3, 3) - point_crossmat * state_propagat.cov.block<3, 3>(0, 0) * point_crossmat;

      // point_body cov
      var = state_propagat.rot_end * extR_ * ptpl_list_[i].body_cov_ * (state_propagat.rot_end * extR_).transpose();

      double sigma_l = J_nq * ptpl_list_[i].plane_var_ * J_nq.transpose();

      if (config_setting_.correspondence_mode_ == "fastlio_ikdtree")
      {
        // FAST-LIO uses the fixed LASER_POINT_COV=0.001 for accepted residuals.
        R_inv(i) = 1.0 / kFastLioLaserPointCovariance;
      }
      else
      {
        R_inv(i) = 1.0 / (0.001 + sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);
        if (lio_diagnostics_.geometryEnabled())
        {
          auto &record = diagnostic_geometry_[ptpl.source_index_].voxel;
          record.update_plane_variance = sigma_l;
          record.update_point_variance = ptpl.normal_.dot(var * ptpl.normal_);
          record.update_variance = 1.0 / R_inv(i);
        }
      }
      // R_inv(i) = 1.0 / (sigma_l + ptpl_list_[i].normal_.transpose() * var * ptpl_list_[i].normal_);

      /*** calculate the Measuremnt Jacobian matrix H ***/
      V3D A(point_crossmat * state_.rot_end.transpose() * ptpl_list_[i].normal_);
      Hsub.row(i) << VEC_FROM_ARRAY(A), ptpl_list_[i].normal_[0], ptpl_list_[i].normal_[1], ptpl_list_[i].normal_[2];
      Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), ptpl_list_[i].normal_[0] * R_inv(i),
          ptpl_list_[i].normal_[1] * R_inv(i), ptpl_list_[i].normal_[2] * R_inv(i);
      meas_vec(i) = -ptpl_list_[i].dis_to_plane_;
    }
    EKF_stop_flg = false;
    flg_EKF_converged = false;
    /*** Iterative Kalman Filter Update ***/
    MatrixXd K(DIM_STATE, effct_feat_num_);
    // auto &&Hsub_T = Hsub.transpose();
    auto &&HTz = Hsub_T_R_inv * meas_vec;
    // fout_dbg<<"HTz: "<<HTz<<endl;
    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
    // IKFoM expresses the propagated covariance in the tangent space of the
    // current iterated state before computing each gain.  This is essential
    // for SO(3), and for gravity whose state lies on S2.
    const auto propagated_to_current_delta = state_ - state_propagat;
    MD(DIM_STATE, DIM_STATE) propagated_to_current = MD(DIM_STATE, DIM_STATE)::Identity();
    propagated_to_current.block<3, 3>(0, 0) =
        so3RetractionJacobian(propagated_to_current_delta.block<3, 1>(0, 0)).transpose();
    if (config_setting_.gravity_norm_constraint_)
    {
      propagated_to_current.block<3, 3>(16, 16) =
          gravityTangentTransport(state_propagat.gravity, state_.gravity);
    }
    MD(DIM_STATE, DIM_STATE) iteration_covariance =
        propagated_to_current * propagated_covariance * propagated_to_current.transpose();
    if (config_setting_.gravity_norm_constraint_ && state_.gravity.norm() > 1e-9)
    {
      const V3D gravity_unit = state_.gravity.normalized();
      iteration_covariance.block<3, 3>(16, 16) += gravity_unit * gravity_unit.transpose() * 1e-8;
    }
    iteration_covariance = 0.5 * (iteration_covariance + iteration_covariance.transpose());
    auto vec = state_propagat - state_;
    vec.block<3, 1>(0, 0) =
        -propagated_to_current.block<3, 3>(0, 0) * propagated_to_current_delta.block<3, 1>(0, 0);
    if (config_setting_.gravity_norm_constraint_ && state_.gravity.norm() > 1e-9)
    {
      const V3D gravity_unit = state_.gravity.normalized();
      vec.block<3, 1>(16, 0) =
          (M3D::Identity() - gravity_unit * gravity_unit.transpose()) * (state_propagat.gravity - state_.gravity);
    }
    VD(DIM_STATE) solution;
    fast_livo2::xicp::Detection xicp_detection;
    fast_livo2::xicp::Update xicp_update;
    MatrixXd diagnostic_baseline_gain;
    const StatesGroup before_update = state_;
    const double xicp_start = omp_get_wtime();
    if (config_setting_.xicp_.enabled)
    {
      try
      {
        MatrixXd local_points(effct_feat_num_, 3), local_normals(effct_feat_num_, 3);
        const M3D lidar_from_world = (state_.rot_end * extR_).transpose();
        for (int i = 0; i < effct_feat_num_; ++i)
        {
          local_points.row(i) = ptpl_list_[i].point_b_.transpose();
          local_normals.row(i) = (lidar_from_world * ptpl_list_[i].normal_).transpose();
        }
        xicp_detection = fast_livo2::xicp::detect(local_points, local_normals, config_setting_.xicp_);
        const MatrixXd local_pose = fast_livo2::xicp::localPoseJacobian(state_.rot_end, extR_, extT_, DIM_STATE);
        xicp_update = fast_livo2::xicp::update(iteration_covariance, Hsub, R_inv, meas_vec, vec,
                                             xicp_detection.constraints * local_pose, xicp_detection.target_gain);
        solution = xicp_update.correction;
      }
      catch (const std::exception &error)
      {
        // Never fall back to an unconstrained update on a numerical failure.
        // Reject this scan's correction, keeping its original IMU prediction.
        RCLCPP_ERROR(rclcpp::get_logger("laserMapping"), "X-ICP rejected LIO update at %.6f: %s", scan_time, error.what());
        state_ = state_propagat;
        captureGeometry(iterCount, state_propagat, before_update, state_);
        if (lio_diagnostics_.enabled()) lio_diagnostics_.endScan(DiagnosticStatus::Rejected, iterCount, state_.cov);
        return;
      }
    }
    else
    {
      // Preserve the pre-X-ICP update path for controlled A/B replay.
      MD(DIM_STATE, DIM_STATE) &&K_1 =
          (H_T_H.block<DIM_STATE, DIM_STATE>(0, 0) + iteration_covariance.inverse()).inverse();
      G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
      solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec.block<DIM_STATE, 1>(0, 0) - G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0);
      if (lio_diagnostics_.enabled())
      {
        try { diagnostic_baseline_gain = K_1.block<DIM_STATE, 6>(0, 0) * Hsub_T_R_inv; }
        catch (const std::exception &error) { lio_diagnostics_.disable(error.what()); }
        catch (...) { lio_diagnostics_.disable("gain extraction failed"); }
      }
    }
    if (config_setting_.gravity_norm_constraint_ && state_.gravity.norm() > 1e-9)
    {
      // FAST-LIO represents gravity on S2. Remove the unobservable radial
      // component before applying this Euclidean error-state correction.
      const V3D gravity_unit = state_.gravity.normalized();
      V3D gravity_delta = solution.block<3, 1>(16, 0);
      gravity_delta -= gravity_unit * gravity_unit.dot(gravity_delta);
      solution.block<3, 1>(16, 0) = gravity_delta;
    }
    const V3D gravity_before_update = state_.gravity;
    state_ += solution;
    constrainGravityState(false);
    if (config_setting_.xicp_.enabled && config_setting_.xicp_.diagnostics)
      logXicp(scan_time, iterCount, xicp_detection, xicp_update, state_propagat, before_update,
              (omp_get_wtime() - xicp_start) * 1000.0);
    // FAST-LIO declares convergence only when every error-state component is
    // within 0.001. This also controls when nearest neighbors are searched again.
    flg_EKF_converged = (solution.array().abs() <= 0.001).all();
    V3D euler_cur = state_.rot_end.eulerAngles(2, 1, 0);

    /*** Rematch Judgement ***/

    const bool force_final_rematch =
        (rematch_num == 0) && (iterCount == (config_setting_.max_iterations_ - 2));
    recompute_fastlio_neighbors = flg_EKF_converged || force_final_rematch;
    if (recompute_fastlio_neighbors) { rematch_num++; }

    /*** Convergence Judgements and Covariance Update ***/
    if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == config_setting_.max_iterations_ - 1)))
    {
      /*** Covariance Update ***/
      // _state.cov = (I_STATE - G) * _state.cov;
      MD(DIM_STATE, DIM_STATE) update_transport = MD(DIM_STATE, DIM_STATE)::Identity();
      update_transport.block<3, 3>(0, 0) = so3RetractionJacobian(solution.block<3, 1>(0, 0)).transpose();
      if (config_setting_.gravity_norm_constraint_)
      {
        update_transport.block<3, 3>(16, 16) = gravityTangentTransport(gravity_before_update, state_.gravity);
      }
      if (config_setting_.xicp_.enabled)
        state_.cov = update_transport * xicp_update.covariance * update_transport.transpose();
      else
        state_.cov = update_transport *
                     (I_STATE.block<DIM_STATE, DIM_STATE>(0, 0) - G.block<DIM_STATE, DIM_STATE>(0, 0)) *
                     iteration_covariance * update_transport.transpose();
      state_.cov = 0.5 * (state_.cov + state_.cov.transpose());
      constrainGravityState(true);
      // total_distance += (_state.pos_end - position_last).norm();
      position_last_ = state_.pos_end;
      geoQuat_ = tf::createQuaternionMsgFromRollPitchYaw(euler_cur(0), euler_cur(1), euler_cur(2));

      // VD(DIM_STATE) K_sum  = K.rowwise().sum();
      // VD(DIM_STATE) P_diag = _state.cov.diagonal();
      EKF_stop_flg = true;
      diagnostic_status = DiagnosticStatus::Updated;
    }
    if (lio_diagnostics_.enabled())
    {
      try
      {
        MatrixXd constraints(0, DIM_STATE), target_gain(0, effct_feat_num_);
        if (config_setting_.xicp_.enabled)
        {
          constraints = xicp_detection.constraints *
              fast_livo2::xicp::localPoseJacobian(before_update.rot_end, extR_, extT_, DIM_STATE);
          target_gain = xicp_detection.target_gain;
        }
        lio_diagnostics_.iteration(iterCount, EKF_stop_flg, Hsub, R_inv, meas_vec, vec, solution,
                                   iteration_covariance,
                                   config_setting_.xicp_.enabled ? xicp_update.gain : diagnostic_baseline_gain,
                                   constraints, target_gain, xicp_update.constraint_error);
        if (EKF_stop_flg) captureGeometry(iterCount, state_propagat, before_update, state_);
      }
      catch (const std::exception &error) { lio_diagnostics_.disable(error.what()); }
      catch (...) { lio_diagnostics_.disable("iteration capture failed"); }
    }
    if (EKF_stop_flg) break;
  }
  if (lio_diagnostics_.enabled()) lio_diagnostics_.endScan(diagnostic_status, diagnostic_last_iteration, state_.cov);

  // double t2 = omp_get_wtime();
  // scan_count++;
  // ekf_time = t2 - t0 - build_residual_time;

  // ave_build_residual_time = ave_build_residual_time * (scan_count - 1) / scan_count + build_residual_time / scan_count;
  // ave_ekf_time = ave_ekf_time * (scan_count - 1) / scan_count + ekf_time / scan_count;

  // cout << "[ Mapping ] ekf_time: " << ekf_time << "s, build_residual_time: " << build_residual_time << "s" << endl;
  // cout << "[ Mapping ] ave_ekf_time: " << ave_ekf_time << "s, ave_build_residual_time: " << ave_build_residual_time << "s" << endl;
}

void VoxelMapManager::captureGeometry(int iteration, const StatesGroup &prior,
                                            const StatesGroup &linearization, const StatesGroup &after) noexcept
{
  if (!lio_diagnostics_.geometryEnabled()) return;
  try
  {
    const auto convert = [](const StatesGroup &state) {
      fast_livo2::diagnostics::GeometryState result;
      result.rotation = state.rot_end; result.position = state.pos_end; result.velocity = state.vel_end;
      result.bg = state.bias_g; result.ba = state.bias_a; result.gravity = state.gravity;
      result.inverse_exposure = state.inv_expo_time;
      return result;
    };
    if (config_setting_.correspondence_mode_ == "fastlio_ikdtree")
    for (size_t i = 0; i < diagnostic_geometry_.size(); ++i)
    {
      auto &point = diagnostic_geometry_[i];
      const auto &neighbors = fastlio_nearest_points_[i];
      point.neighbor_count = static_cast<int>(neighbors.size());
      for (int j = 0; j < std::min(point.neighbor_count, 5); ++j)
        point.neighbors[j] = V3D(neighbors[j].x, neighbors[j].y, neighbors[j].z);
    }
    lio_diagnostics_.geometry(iteration, convert(prior), convert(linearization), convert(after),
                              extR_, extT_, diagnostic_geometry_);
  }
  catch (const std::exception &error) { lio_diagnostics_.disable(error.what()); }
  catch (...) { lio_diagnostics_.disable("geometry preparation failed"); }
}

void VoxelMapManager::constrainGravityState(const bool project_covariance)
{
  if (!config_setting_.gravity_norm_constraint_) return;

  const double gravity_norm = state_.gravity.norm();
  if (!std::isfinite(gravity_norm) || gravity_norm < 1e-9) return;

  const V3D gravity_unit = state_.gravity / gravity_norm;
  state_.gravity = gravity_unit * G_m_s2;
  if (!project_covariance) return;

  // Embed S2's two-dimensional tangent covariance in the existing 3D state.
  // A tiny radial variance keeps the legacy 19x19 information matrix invertible.
  const M3D tangent_projector = M3D::Identity() - gravity_unit * gravity_unit.transpose();
  MD(DIM_STATE, DIM_STATE) state_projector = MD(DIM_STATE, DIM_STATE)::Identity();
  state_projector.block<3, 3>(16, 16) = tangent_projector;
  state_.cov = state_projector * state_.cov * state_projector.transpose();
  state_.cov.block<3, 3>(16, 16) += gravity_unit * gravity_unit.transpose() * 1e-8;
  state_.cov = 0.5 * (state_.cov + state_.cov.transpose());
}

M3D VoxelMapManager::boundedStateCovarianceBlock(const int start_index) const
{
  const M3D covariance = 0.5 * (state_.cov.block<3, 3>(start_index, start_index) +
                                state_.cov.block<3, 3>(start_index, start_index).transpose());
  Eigen::SelfAdjointEigenSolver<M3D> solver(covariance);
  if (solver.info() != Eigen::Success)
  {
    return M3D::Identity() * config_setting_.max_state_cov_for_matching_;
  }

  const V3D eigenvalues = solver.eigenvalues().cwiseMax(0.0).cwiseMin(config_setting_.max_state_cov_for_matching_);
  return solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
}

void VoxelMapManager::TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                                     pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud)
{
  pcl::PointCloud<pcl::PointXYZI>().swap(*trans_cloud);
  trans_cloud->reserve(input_cloud->size());
  for (size_t i = 0; i < input_cloud->size(); i++)
  {
    pcl::PointXYZINormal p_c = input_cloud->points[i];
    Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
    p = (rot * (extR_ * p + extT_) + t);
    pcl::PointXYZI pi;
    pi.x = p(0);
    pi.y = p(1);
    pi.z = p(2);
    pi.intensity = p_c.intensity;
    trans_cloud->points.push_back(pi);
  }
}

void VoxelMapManager::BuildVoxelMap()
{
  buildVoxelMapFromPoints(initialMapPoints());
  BuildFastLioMap();
}

std::vector<pointWithVar> VoxelMapManager::initialMapPoints() const
{
  std::vector<pointWithVar> input_points;
  input_points.reserve(feats_down_world_->size());
  const auto pose_cov=fast_livo2::boundedPoseCovariance(state_.cov.topLeftCorner<6,6>(),
      config_setting_.max_state_cov_for_matching_);

  for (size_t i = 0; i < feats_down_world_->size(); i++)
  {
    pointWithVar pv;
    pv.point_w << feats_down_world_->points[i].x, feats_down_world_->points[i].y, feats_down_world_->points[i].z;
    V3D point_this(feats_down_body_->points[i].x, feats_down_body_->points[i].y, feats_down_body_->points[i].z);
    M3D var;
    calcBodyCov(point_this, config_setting_.dept_err_, config_setting_.beam_err_, var);
    var=fast_livo2::worldPointCovariance(extR_*point_this+extT_,state_.rot_end,extR_,var,pose_cov);
    pv.var = var;
    input_points.push_back(pv);
  }
  return input_points;
}

void VoxelMapManager::buildVoxelMapFromPoints(const std::vector<pointWithVar> &input_points)
{
  const float voxel_size = config_setting_.max_voxel_size_;
  const float planer_threshold = config_setting_.planner_threshold_;
  const int max_layer = config_setting_.max_layer_;
  const int max_points_num = config_setting_.max_points_num_;
  const auto &layer_init_num = config_setting_.layer_init_num_;
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    const VOXEL_LOCATION position = worldToVoxelLocation(p_v.point_w, voxel_size);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end())
    {
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
    }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->temp_points_.push_back(p_v);
      voxel_map_[position]->new_points_++;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
    }
  }
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); ++iter)
  {
    iter->second->init_octo_tree();
  }
}

V3F VoxelMapManager::RGBFromVoxel(const V3D &input_point)
{
  const VOXEL_LOCATION position = worldToVoxelLocation(input_point, config_setting_.max_voxel_size_);
  int64_t ind = position.x + position.y + position.z;
  uint k((ind + 100000) % 3);
  V3F RGB((k == 0) * 255.0, (k == 1) * 255.0, (k == 2) * 255.0);
  // cout<<"RGB: "<<RGB.transpose()<<endl;
  return RGB;
}

void VoxelMapManager::UpdateVoxelMap(const std::vector<pointWithVar> &input_points)
{
  float voxel_size = config_setting_.max_voxel_size_;
  float planer_threshold = config_setting_.planner_threshold_;
  int max_layer = config_setting_.max_layer_;
  int max_points_num = config_setting_.max_points_num_;
  std::vector<int> layer_init_num = config_setting_.layer_init_num_;
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++)
  {
    const pointWithVar p_v = input_points[i];
    const VOXEL_LOCATION position = worldToVoxelLocation(p_v.point_w, voxel_size);
    auto iter = voxel_map_.find(position);
    if (iter != voxel_map_.end()) { voxel_map_[position]->UpdateOctoTree(p_v); }
    else
    {
      VoxelOctoTree *octo_tree = new VoxelOctoTree(max_layer, 0, layer_init_num[0], max_points_num, planer_threshold);
      voxel_map_[position] = octo_tree;
      voxel_map_[position]->layer_init_num_ = layer_init_num;
      voxel_map_[position]->quater_length_ = voxel_size / 4;
      voxel_map_[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      voxel_map_[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      voxel_map_[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      voxel_map_[position]->UpdateOctoTree(p_v);
    }
  }
}

bool VoxelMapManager::FastLioMapReady() const
{
  return config_setting_.correspondence_mode_ == "fastlio_ikdtree" && fastlio_ikdtree_.Root_Node != nullptr;
}

void VoxelMapManager::BuildFastLioMap()
{
  if (config_setting_.correspondence_mode_ != "fastlio_ikdtree" ||
      feats_down_world_ == nullptr || feats_down_world_->empty() || fastlio_ikdtree_.Root_Node != nullptr)
  {
    return;
  }

  PointVector initial_points;
  initial_points.reserve(feats_down_world_->size());
  for (const auto &point : feats_down_world_->points)
  {
    if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) initial_points.push_back(point);
  }
  if (initial_points.size() <= static_cast<size_t>(kFastLioNumMatchPoints)) return;

  fastlio_ikdtree_.set_downsample_param(static_cast<float>(config_setting_.fastlio_filter_size_map_));
  fastlio_ikdtree_.Build(initial_points);
  SegmentFastLioLocalMap();
  std::cout << "[ LIO ] Initialized FAST-LIO ikd-tree with " << fastlio_ikdtree_.validnum() << " points" << std::endl;
}

void VoxelMapManager::SegmentFastLioLocalMap()
{
  const V3D lidar_position = state_.pos_end + state_.rot_end * extT_;
  const double cube_side_length = config_setting_.fastlio_cube_side_length_;
  const double detection_range = config_setting_.fastlio_detection_range_;

  if (!fastlio_local_map_initialized_)
  {
    for (int axis = 0; axis < 3; ++axis)
    {
      fastlio_local_map_points_.vertex_min[axis] = static_cast<float>(lidar_position(axis) - cube_side_length * 0.5);
      fastlio_local_map_points_.vertex_max[axis] = static_cast<float>(lidar_position(axis) + cube_side_length * 0.5);
    }
    fastlio_local_map_initialized_ = true;
    return;
  }

  double distance_to_edge[3][2];
  bool move_required = false;
  for (int axis = 0; axis < 3; ++axis)
  {
    distance_to_edge[axis][0] = std::abs(lidar_position(axis) - fastlio_local_map_points_.vertex_min[axis]);
    distance_to_edge[axis][1] = std::abs(lidar_position(axis) - fastlio_local_map_points_.vertex_max[axis]);
    if (distance_to_edge[axis][0] <= kFastLioMoveThreshold * detection_range ||
        distance_to_edge[axis][1] <= kFastLioMoveThreshold * detection_range)
    {
      move_required = true;
    }
  }
  if (!move_required || fastlio_ikdtree_.Root_Node == nullptr) return;

  BoxPointType new_local_map = fastlio_local_map_points_;
  std::vector<BoxPointType> boxes_to_remove;
  const double move_distance =
      std::max((cube_side_length - 2.0 * kFastLioMoveThreshold * detection_range) * 0.5 * 0.9,
               detection_range * (kFastLioMoveThreshold - 1.0));

  for (int axis = 0; axis < 3; ++axis)
  {
    BoxPointType removal_box = fastlio_local_map_points_;
    if (distance_to_edge[axis][0] <= kFastLioMoveThreshold * detection_range)
    {
      new_local_map.vertex_min[axis] -= static_cast<float>(move_distance);
      new_local_map.vertex_max[axis] -= static_cast<float>(move_distance);
      removal_box.vertex_min[axis] = fastlio_local_map_points_.vertex_max[axis] - static_cast<float>(move_distance);
      boxes_to_remove.push_back(removal_box);
    }
    else if (distance_to_edge[axis][1] <= kFastLioMoveThreshold * detection_range)
    {
      new_local_map.vertex_min[axis] += static_cast<float>(move_distance);
      new_local_map.vertex_max[axis] += static_cast<float>(move_distance);
      removal_box.vertex_max[axis] = fastlio_local_map_points_.vertex_min[axis] + static_cast<float>(move_distance);
      boxes_to_remove.push_back(removal_box);
    }
  }
  fastlio_local_map_points_ = new_local_map;

  PointVector removed_points;
  fastlio_ikdtree_.acquire_removed_points(removed_points);
  if (!boxes_to_remove.empty()) fastlio_ikdtree_.Delete_Point_Boxes(boxes_to_remove);
}

void VoxelMapManager::UpdateFastLioMap(const PointCloudXYZI::Ptr &world_cloud)
{
  if (config_setting_.correspondence_mode_ != "fastlio_ikdtree" || world_cloud == nullptr || world_cloud->empty()) return;
  if (fastlio_ikdtree_.Root_Node == nullptr)
  {
    BuildFastLioMap();
    return;
  }

  PointVector points_to_add;
  PointVector points_without_downsampling;
  points_to_add.reserve(world_cloud->size());
  points_without_downsampling.reserve(world_cloud->size());
  const float filter_size = static_cast<float>(config_setting_.fastlio_filter_size_map_);

  for (size_t i = 0; i < world_cloud->size(); ++i)
  {
    const PointType &point = world_cloud->points[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;

    bool need_add = true;
    if (i < fastlio_nearest_points_.size() && !fastlio_nearest_points_[i].empty())
    {
      const PointVector &nearby_points = fastlio_nearest_points_[i];
      PointType voxel_center;
      voxel_center.x = std::floor(point.x / filter_size) * filter_size + 0.5f * filter_size;
      voxel_center.y = std::floor(point.y / filter_size) * filter_size + 0.5f * filter_size;
      voxel_center.z = std::floor(point.z / filter_size) * filter_size + 0.5f * filter_size;
      const auto squared_distance = [](const PointType &lhs, const PointType &rhs) {
        const float dx = lhs.x - rhs.x;
        const float dy = lhs.y - rhs.y;
        const float dz = lhs.z - rhs.z;
        return dx * dx + dy * dy + dz * dz;
      };
      const float candidate_distance = squared_distance(point, voxel_center);

      if (std::abs(nearby_points[0].x - voxel_center.x) > 0.5f * filter_size &&
          std::abs(nearby_points[0].y - voxel_center.y) > 0.5f * filter_size &&
          std::abs(nearby_points[0].z - voxel_center.z) > 0.5f * filter_size)
      {
        points_without_downsampling.push_back(point);
        continue;
      }

      for (int neighbor = 0; neighbor < kFastLioNumMatchPoints; ++neighbor)
      {
        if (nearby_points.size() < static_cast<size_t>(kFastLioNumMatchPoints)) break;
        if (squared_distance(nearby_points[neighbor], voxel_center) < candidate_distance)
        {
          need_add = false;
          break;
        }
      }
    }
    if (need_add) points_to_add.push_back(point);
  }

  fastlio_ikdtree_.Add_Points(points_to_add, true);
  fastlio_ikdtree_.Add_Points(points_without_downsampling, false);
}

bool VoxelMapManager::estimateFastLioPlane(const PointVector &points, V3D &normal, double &d, V3D &center) const
{
  if (points.size() != static_cast<size_t>(kFastLioNumMatchPoints)) return false;

  Eigen::Matrix<double, kFastLioNumMatchPoints, 3> coordinates;
  center.setZero();
  for (int i = 0; i < kFastLioNumMatchPoints; ++i)
  {
    const PointType &point = points[i];
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) return false;
    coordinates.row(i) << point.x, point.y, point.z;
    center += V3D(point.x, point.y, point.z);
  }
  center /= static_cast<double>(kFastLioNumMatchPoints);

  const Eigen::Matrix<double, kFastLioNumMatchPoints, 1> rhs =
      Eigen::Matrix<double, kFastLioNumMatchPoints, 1>::Constant(-1.0);
  const V3D unnormalized_normal = coordinates.colPivHouseholderQr().solve(rhs);
  const double normal_norm = unnormalized_normal.norm();
  if (!std::isfinite(normal_norm) || normal_norm < 1e-9) return false;

  normal = unnormalized_normal / normal_norm;
  d = 1.0 / normal_norm;
  for (int i = 0; i < kFastLioNumMatchPoints; ++i)
  {
    if (std::abs(normal.dot(coordinates.row(i).transpose()) + d) > kFastLioPlaneThreshold) return false;
  }
  return true;
}

void VoxelMapManager::BuildResidualListFastLio(std::vector<pointWithVar> &pv_list,
                                                std::vector<PointToPlane> &ptpl_list,
                                                const bool recompute_neighbors, const int iteration)
{
  using MatchStatus = fast_livo2::diagnostics::MatchStatus;
  const bool capture = lio_diagnostics_.geometryEnabled();
  const bool evaluate_normals = !source_normals_.empty();
  normal_stats_.comparisons = normal_stats_.rejected = 0;
  lio_diagnostics_.normalStats(normal_stats_);
  ptpl_list.clear();
  if (fastlio_ikdtree_.validnum() < kFastLioNumMatchPoints)
  {
    if (capture) for (auto &point : diagnostic_geometry_) point.status = MatchStatus::InsufficientNeighbors;
    return;
  }

  std::vector<uint8_t> compared(evaluate_normals ? pv_list.size() : 0, 0);
  std::vector<uint8_t> rejected(evaluate_normals ? pv_list.size() : 0, 0);
  std::vector<PointToPlane> candidates(pv_list.size());
  std::vector<uint8_t> useful(pv_list.size(), 0);

  #ifdef MP_EN
    omp_set_num_threads(std::getenv("FAST_LIVO2_DETERMINISTIC") != nullptr ? 1 : MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < static_cast<int>(pv_list.size()); ++i)
  {
    pointWithVar &pv = pv_list[i];
    auto *record = capture ? &diagnostic_geometry_[i] : nullptr;
    if (!pv.point_b.allFinite() || !pv.point_w.allFinite())
    {
      fastlio_neighbor_valid_[i] = 0;
      fastlio_nearest_points_[i].clear();
      continue;
    }

    PointType query;
    query.x = static_cast<float>(pv.point_w.x());
    query.y = static_cast<float>(pv.point_w.y());
    query.z = static_cast<float>(pv.point_w.z());

    if (recompute_neighbors)
    {
      std::vector<float> squared_distances;
      fastlio_nearest_points_[i].clear();
      fastlio_ikdtree_.Nearest_Search(query, kFastLioNumMatchPoints,
                                      fastlio_nearest_points_[i], squared_distances);
      fastlio_neighbor_valid_[i] =
          fastlio_nearest_points_[i].size() == static_cast<size_t>(kFastLioNumMatchPoints) &&
          squared_distances.size() == static_cast<size_t>(kFastLioNumMatchPoints) &&
          squared_distances[kFastLioNumMatchPoints - 1] <= kFastLioMaxNeighborSquaredDistance;
      if (record)
      {
        record->search_world = V3D(query.x, query.y, query.z);
        record->search_iteration = iteration;
        record->search_farthest_squared = squared_distances.empty() ? 0.0 : squared_distances.back();
      }
    }
    if (fastlio_neighbor_valid_[i] == 0)
    {
      if (record) record->status = fastlio_nearest_points_[i].size() < kFastLioNumMatchPoints ?
          MatchStatus::InsufficientNeighbors : MatchStatus::NeighborDistance;
      continue;
    }

    V3D normal;
    V3D center;
    double d = 0.0;
    if (!estimateFastLioPlane(fastlio_nearest_points_[i], normal, d, center))
    {
      if (record) record->status = MatchStatus::PlaneFit;
      continue;
    }

    const double signed_distance = normal.dot(pv.point_w) + d;
    const double body_range = pv.point_b.norm();
    if (record)
    {
      record->normal = normal; record->center = center; record->plane_d = d;
      record->distance = signed_distance; record->status = MatchStatus::Residual;
    }
    if (!std::isfinite(signed_distance) || !std::isfinite(body_range) || body_range <= 1e-9) continue;
    const double score = 1.0 - 0.9 * std::abs(signed_distance) / std::sqrt(body_range);
    if (score <= kFastLioResidualScoreThreshold) continue;

    if (evaluate_normals)
    {
      // Reevaluate at every iteration even when the five map neighbors are cached.
      const auto comparison = fast_livo2::normal_consistency::compare(
          source_normals_[i], normal, state_.rot_end, extR_, config_setting_.normal_consistency_);
      compared[i] = comparison.valid;
      if (record) record->comparison = comparison;
      if (config_setting_.normal_consistency_.enabled && comparison.mismatch)
      {
        rejected[i] = 1;
        if (record) record->status = MatchStatus::NormalMismatch;
        continue;
      }
    }

    PointToPlane &match = candidates[i];
    match.point_b_ = pv.point_b;
    match.point_w_ = pv.point_w;
    match.normal_ = normal;
    match.center_ = center;
    match.plane_var_.setZero();
    match.body_cov_ = pv.body_var;
    match.layer_ = 0;
    match.d_ = d;
    match.eigen_value_ = 0.0;
    match.is_valid_ = true;
    match.dis_to_plane_ = static_cast<float>(signed_distance);
    pv.normal = normal;
    useful[i] = 1;
    if (record) record->status = MatchStatus::Accepted;
  }

  for (const auto value : compared) normal_stats_.comparisons += value;
  for (const auto value : rejected) normal_stats_.rejected += value;
  lio_diagnostics_.normalStats(normal_stats_);
  ptpl_list.reserve(pv_list.size());
  for (size_t i = 0; i < useful.size(); ++i)
  {
    if (useful[i] != 0)
    {
      if (capture) diagnostic_geometry_[i].match_index = static_cast<int>(ptpl_list.size());
      ptpl_list.push_back(candidates[i]);
    }
  }
}

void VoxelMapManager::BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list)
{
  using fast_livo2::diagnostics::MatchStatus;
  const bool capture = lio_diagnostics_.geometryEnabled();
  double voxel_size = config_setting_.max_voxel_size_;
  const int neighbor_radius = config_setting_.neighbor_voxel_radius_;
  std::mutex mylock;
  ptpl_list.clear();
  std::vector<PointToPlane> all_ptpl_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());
  for (size_t i = 0; i < index.size(); ++i)
  {
    index[i] = i;
    useful_ptpl[i] = false;
  }
  #ifdef MP_EN
    omp_set_num_threads(std::getenv("FAST_LIVO2_DETERMINISTIC") != nullptr ? 1 : MP_PROC_NUM);
    #pragma omp parallel for
  #endif
  for (int i = 0; i < index.size(); i++)
  {
    pointWithVar &pv = pv_list[i];
    auto *record = capture ? &diagnostic_geometry_[i] : nullptr;
    if (record)
    {
      record->voxel.point_world_covariance = pv.var;
      record->voxel.point_body_covariance = pv.body_var;
    }
    if (!pv.point_b.allFinite() || !pv.point_w.allFinite()) continue;
    if (record) record->status = MatchStatus::NoVoxel;
    const VOXEL_LOCATION position = worldToVoxelLocation(pv.point_w, voxel_size);
    PointToPlane single_ptpl;
    bool is_sucess = false;
    double prob = 0;

    // Search the current voxel and every voxel in the configured neighborhood.
    // This keeps data association continuous across voxel boundaries and also
    // allows recovery when the IMU prediction lands in an empty voxel.
    for (int dx = -neighbor_radius; dx <= neighbor_radius; ++dx)
    {
      for (int dy = -neighbor_radius; dy <= neighbor_radius; ++dy)
      {
        for (int dz = -neighbor_radius; dz <= neighbor_radius; ++dz)
        {
          const VOXEL_LOCATION candidate(position.x + dx, position.y + dy, position.z + dz);
          const auto iter = voxel_map_.find(candidate);
          if (iter != voxel_map_.end())
          {
            if (record && record->status == MatchStatus::NoVoxel) record->status = MatchStatus::NoPlane;
            build_single_residual(pv, iter->second, 0, is_sucess, prob, single_ptpl, record, candidate);
          }
        }
      }
    }

    if (is_sucess)
    {
      mylock.lock();
      useful_ptpl[i] = true;
      all_ptpl_list[i] = single_ptpl;
      mylock.unlock();
    }
    else
    {
      mylock.lock();
      useful_ptpl[i] = false;
      mylock.unlock();
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++)
  {
    if (useful_ptpl[i])
    {
      all_ptpl_list[i].source_index_ = static_cast<int>(i);
      if (capture) diagnostic_geometry_[i].match_index = static_cast<int>(ptpl_list.size());
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

void VoxelMapManager::build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess,
                                            double &prob, PointToPlane &single_ptpl,
                                            fast_livo2::diagnostics::GeometryPoint *record, const VOXEL_LOCATION &root_voxel)
{
  using fast_livo2::diagnostics::MatchStatus;
  int max_layer = config_setting_.max_layer_;
  double sigma_num = config_setting_.sigma_num_;

  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_w;
  if (current_octo->plane_ptr_->is_plane_)
  {
    VoxelPlane &plane = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center_;
    float dis_to_plane = fabs(plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_);
    float dis_to_center = (plane.center_(0) - p_w(0)) * (plane.center_(0) - p_w(0)) + (plane.center_(1) - p_w(1)) * (plane.center_(1) - p_w(1)) +
                          (plane.center_(2) - p_w(2)) * (plane.center_(2) - p_w(2));
    // Roundoff at the plane center must not turn a zero tangential distance into NaN.
    float range_dis = sqrt(std::max(0.0f, dis_to_center - dis_to_plane * dis_to_plane));
    const auto record_candidate = [&](MatchStatus status, double plane_variance, double point_variance, double probability) {
      if (!record) return;
      const auto stage = [](MatchStatus s) {
        switch (s)
        {
        case MatchStatus::Accepted: return 4;
        case MatchStatus::Residual: return 3;
        case MatchStatus::InvalidVariance: return 2;
        case MatchStatus::OutsideExtent: return 1;
        default: return 0;
        }
      };
      const int next = stage(status), previous = stage(record->status);
      if (next < previous) return;
      if (next == previous && status != MatchStatus::Accepted)
      {
        if (status == MatchStatus::Residual && dis_to_plane / std::sqrt(plane_variance + point_variance) >=
              record->voxel.gate_distance / std::sqrt(record->voxel.gate_variance)) return;
        if (status == MatchStatus::OutsideExtent && range_dis / std::max(1e-15f, plane.radius_) >=
              record->voxel.tangential_distance / std::max(1e-15, record->voxel.radius)) return;
        if (status == MatchStatus::InvalidVariance) return;
      }
      record->status = status;
      record->normal = plane.normal_; record->center = plane.center_;
      record->plane_d = plane.d_;
      record->distance = plane.normal_.dot(p_w) + plane.d_;
      auto &v = record->voxel;
      v.plane_valid = true; v.plane_id = plane.id_; v.layer = current_layer;
      v.support_points = plane.points_size_; v.update_enabled = current_octo->update_enable_;
      v.root_voxel = {root_voxel.x, root_voxel.y, root_voxel.z};
      v.voxel_center = V3D(current_octo->voxel_center_[0], current_octo->voxel_center_[1], current_octo->voxel_center_[2]);
      v.eigenvalues = V3D(plane.min_eigen_value_, plane.mid_eigen_value_, plane.max_eigen_value_);
      v.plane_covariance = plane.plane_var_; v.radius = plane.radius_;
      v.tangential_distance = range_dis; v.gate_distance = dis_to_plane;
      v.gate_plane_variance = plane_variance; v.gate_point_variance = point_variance;
      v.gate_variance = plane_variance + point_variance; v.probability = probability;
    };

    if (range_dis <= radius_k * plane.radius_)
    {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center_;
      J_nq.block<1, 3>(0, 3) = -plane.normal_;
      double sigma_l = J_nq * plane.plane_var_ * J_nq.transpose();
      const double plane_variance = sigma_l;
      const double point_variance = plane.normal_.transpose() * pv.var * plane.normal_;
      sigma_l += point_variance;
      if (!std::isfinite(sigma_l) || sigma_l <= 0.0 || !std::isfinite(dis_to_plane))
      {
        record_candidate(MatchStatus::InvalidVariance, plane_variance, point_variance, 0.0);
        return;
      }
      if (dis_to_plane < sigma_num * sqrt(sigma_l))
      {
        double this_prob = 1.0 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (std::isfinite(this_prob) && this_prob > prob)
        {
          is_sucess = true;
          prob = this_prob;
          pv.normal = plane.normal_;
          single_ptpl.body_cov_ = pv.body_var;
          single_ptpl.point_b_ = pv.point_b;
          single_ptpl.point_w_ = pv.point_w;
          single_ptpl.plane_var_ = plane.plane_var_;
          single_ptpl.normal_ = plane.normal_;
          single_ptpl.center_ = plane.center_;
          single_ptpl.d_ = plane.d_;
          single_ptpl.layer_ = current_layer;
          single_ptpl.dis_to_plane_ = plane.normal_(0) * p_w(0) + plane.normal_(1) * p_w(1) + plane.normal_(2) * p_w(2) + plane.d_;
          record_candidate(MatchStatus::Accepted, plane_variance, point_variance, this_prob);
        }
        return;
      }
      else
      {
        record_candidate(MatchStatus::Residual, plane_variance, point_variance, 0.0);
        return;
      }
    }
    else
    {
      record_candidate(MatchStatus::OutsideExtent, 0.0, 0.0, 0.0);
      return;
    }
  }
  else
  {
    if (current_layer < max_layer)
    {
      for (size_t leafnum = 0; leafnum < 8; leafnum++)
      {
        if (current_octo->leaves_[leafnum] != nullptr)
        {

          VoxelOctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_residual(pv, leaf_octo, current_layer + 1, is_sucess, prob, single_ptpl, record, root_voxel);
        }
      }
      return;
    }
    else { return; }
  }
}

void VoxelMapManager::pubVoxelMap()
{
  double max_trace = 0.25;
  double pow_num = 0.2;
  rclcpp::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<VoxelPlane> pub_plane_list;
  for (auto iter = voxel_map_.begin(); iter != voxel_map_.end(); iter++)
  {
    GetUpdatePlane(iter->second, config_setting_.max_layer_, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++)
  {
    V3D plane_cov = pub_plane_list[i].plane_var_.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) { trace = max_trace; }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane_) { alpha = use_alpha; }
    else { alpha = 0; }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  voxel_map_pub_.publish(voxel_plane);
  loop.sleep();
}

void VoxelMapManager::GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list)
{
  if (current_octo->layer_ > pub_max_voxel_layer) { return; }
  if (current_octo->plane_ptr_->is_update_) { plane_list.push_back(*current_octo->plane_ptr_); }
  if (current_octo->layer_ < current_octo->max_layer_)
  {
    if (!current_octo->plane_ptr_->is_plane_)
    {
      for (size_t i = 0; i < 8; i++)
      {
        if (current_octo->leaves_[i] != nullptr) { GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer, plane_list); }
      }
    }
  }
  return;
}

void VoxelMapManager::pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane,
                                     const float alpha, const Eigen::Vector3d rgb)
{
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = fast_livo2::stampFromSec(0.0);
  plane.ns = plane_ns;
  plane.id = single_plane.id_;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center_[0];
  plane.pose.position.y = single_plane.center_[1];
  plane.pose.position.z = single_plane.center_[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal_, single_plane.y_normal_, single_plane.normal_, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value_);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value_);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value_);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = fast_livo2::durationFromSec(0.0);
  plane_pub.markers.push_back(plane);
}

void VoxelMapManager::CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                                      geometry_msgs::Quaternion &q)
{
  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void VoxelMapManager::mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b)
{
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) { v = vmin; }

  if (v > vmax) { v = vmax; }

  double dr, dg, db;

  if (v < 0.1242)
  {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  }
  else if (v < 0.3747)
  {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  }
  else if (v < 0.6253)
  {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  }
  else if (v < 0.8758)
  {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  }
  else
  {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}

void VoxelMapManager::mapSliding()
{
  if((position_last_ - last_slide_position).norm() < config_setting_.sliding_thresh)
  {
    std::cout<<RED<<"[DEBUG]: Last sliding length "<<(position_last_ - last_slide_position).norm()<<RESET<<"\n";
    return;
  }

  //get global id now
  last_slide_position = position_last_;
  double t_sliding_start = omp_get_wtime();
  const VOXEL_LOCATION position = worldToVoxelLocation(position_last_, config_setting_.max_voxel_size_);
  clearMemOutOfMap(position.x + config_setting_.half_map_size, position.x - config_setting_.half_map_size,
                    position.y + config_setting_.half_map_size, position.y - config_setting_.half_map_size,
                    position.z + config_setting_.half_map_size, position.z - config_setting_.half_map_size);
  double t_sliding_end = omp_get_wtime();
  std::cout<<RED<<"[DEBUG]: Map sliding using "<<t_sliding_end - t_sliding_start<<" secs"<<RESET<<"\n";
  return;
}

void VoxelMapManager::clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min )
{
  int delete_voxel_cout = 0;
  // double delete_time = 0;
  // double last_delete_time = 0;
  for (auto it = voxel_map_.begin(); it != voxel_map_.end(); )
  {
    const VOXEL_LOCATION& loc = it->first;
    bool should_remove = loc.x > x_max || loc.x < x_min || loc.y > y_max || loc.y < y_min || loc.z > z_max || loc.z < z_min;
    if (should_remove){
      // last_delete_time = omp_get_wtime();
      delete it->second;
      it = voxel_map_.erase(it);
      // delete_time += omp_get_wtime() - last_delete_time;
      delete_voxel_cout++;
    } else {
      ++it;
    }
  }
  std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" root voxels"<<RESET<<"\n";
  // std::cout<<RED<<"[DEBUG]: Delete "<<delete_voxel_cout<<" voxels using "<<delete_time<<" s"<<RESET<<"\n";
}
