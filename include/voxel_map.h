/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef VOXEL_MAP_H_
#define VOXEL_MAP_H_

#include "common_lib.h"
#include "ikd-Tree/ikd_Tree.h"
#include "xicp.h"
#include "lio_diagnostics.h"
#include "normal_consistency.h"
#include <Eigen/Dense>
#include <fstream>
#include <math.h>
#include <mutex>
#include <omp.h>
#include <pcl/common/io.h>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#define VOXELMAP_HASH_P 116101
#define VOXELMAP_MAX_N 10000000000

static int voxel_plane_id = 0;

struct VoxelMapInitializationConfig
{
  int min_points = 5000;
  int min_planes = 100;
  // Smallest eigenvalue of mean(n * n.transpose()), where n is a unit normal.
  // Opposite normals count as the same direction; the eigenvalues sum to one.
  double min_normal_eigenvalue = 0.02;
  double min_duration_sec = 0.1;
  double max_duration_sec = 2.0;
  int max_points = 100000;
};

struct VoxelMapInitializationStatus
{
  size_t points = 0;
  size_t scans = 0;
  size_t planes = 0;
  size_t resets = 0;
  double duration_sec = 0.0;
  V3D normal_eigenvalues = V3D::Zero();
  bool ready = false;
};

typedef struct VoxelMapConfig
{
  double max_voxel_size_;
  int max_layer_;
  int max_iterations_;
  std::vector<int> layer_init_num_;
  int max_points_num_;
  double planner_threshold_;
  double beam_err_;
  double dept_err_;
  double sigma_num_;
  double max_state_cov_for_matching_;
  int neighbor_voxel_radius_;
  std::string correspondence_mode_;
  double fastlio_filter_size_map_;
  double fastlio_cube_side_length_;
  double fastlio_detection_range_;
  bool gravity_norm_constraint_;
  bool is_pub_plane_map_;
  VoxelMapInitializationConfig initialization_;
  fast_livo2::xicp::Config xicp_;
  fast_livo2::diagnostics::Config diagnostics_;
  fast_livo2::normal_consistency::Config normal_consistency_;
  std::string debug_log_dir_;

  // config of local map sliding
  double sliding_thresh;
  bool map_sliding_en;
  int half_map_size;
} VoxelMapConfig;

typedef struct PointToPlane
{
  int source_index_ = -1;
  Eigen::Vector3d point_b_;
  Eigen::Vector3d point_w_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d center_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  M3D body_cov_;
  int layer_;
  double d_;
  double eigen_value_;
  bool is_valid_;
  float dis_to_plane_;
} PointToPlane;

typedef struct VoxelPlane
{
  Eigen::Vector3d center_;
  Eigen::Vector3d normal_;
  Eigen::Vector3d y_normal_;
  Eigen::Vector3d x_normal_;
  Eigen::Matrix3d covariance_;
  Eigen::Matrix<double, 6, 6> plane_var_;
  float radius_ = 0;
  float min_eigen_value_ = 1;
  float mid_eigen_value_ = 1;
  float max_eigen_value_ = 1;
  float d_ = 0;
  int points_size_ = 0;
  bool is_plane_ = false;
  bool is_init_ = false;
  int id_ = 0;
  bool is_update_ = false;
  VoxelPlane()
  {
    plane_var_ = Eigen::Matrix<double, 6, 6>::Zero();
    covariance_ = Eigen::Matrix3d::Zero();
    center_ = Eigen::Vector3d::Zero();
    normal_ = Eigen::Vector3d::Zero();
  }
} VoxelPlane;

class VOXEL_LOCATION
{
public:
  int64_t x, y, z;

  VOXEL_LOCATION(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VOXEL_LOCATION &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std
{
template <> struct hash<VOXEL_LOCATION>
{
  int64_t operator()(const VOXEL_LOCATION &s) const
  {
    using std::hash;
    using std::size_t;
    return ((((s.z) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.y)) * VOXELMAP_HASH_P) % VOXELMAP_MAX_N + (s.x);
  }
};
} // namespace std

struct DS_POINT
{
  float xyz[3];
  float intensity;
  int count = 0;
};

void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov);

class VoxelOctoTree
{

public:
  VoxelOctoTree() = default;
  std::vector<pointWithVar> temp_points_;
  VoxelPlane *plane_ptr_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  VoxelOctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  std::vector<int> layer_init_num_;
  float quater_length_;
  float planer_threshold_;
  int points_size_threshold_;
  int update_size_threshold_;
  int max_points_num_;
  int max_layer_;
  int new_points_;
  bool init_octo_;
  bool update_enable_;

  VoxelOctoTree(int max_layer, int layer, int points_size_threshold, int max_points_num, float planer_threshold)
      : max_layer_(max_layer), layer_(layer), points_size_threshold_(points_size_threshold), max_points_num_(max_points_num),
        planer_threshold_(planer_threshold)
  {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_ = 0;
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    for (int i = 0; i < 8; i++)
    {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new VoxelPlane;
  }

  ~VoxelOctoTree()
  {
    for (int i = 0; i < 8; i++)
    {
      delete leaves_[i];
    }
    delete plane_ptr_;
  }
  void init_plane(const std::vector<pointWithVar> &points, VoxelPlane *plane);
  void init_octo_tree();
  void cut_octo_tree();
  void UpdateOctoTree(const pointWithVar &pv);

  VoxelOctoTree *find_correspond(Eigen::Vector3d pw);
  VoxelOctoTree *Insert(const pointWithVar &pv);
};

void loadVoxelConfig(const rclcpp::Node::SharedPtr &node, VoxelMapConfig &voxel_config);

class VoxelMapManager
{
public:
  VoxelMapManager() = default;
  VoxelMapConfig config_setting_;
  int current_frame_id_ = 0;
  fast_livo2::Publisher voxel_map_pub_;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> voxel_map_;

  PointCloudXYZI::Ptr feats_undistort_;
  PointCloudXYZI::Ptr feats_down_body_;
  PointCloudXYZI::Ptr feats_down_world_;

  M3D extR_;
  V3D extT_;
  float build_residual_time, ekf_time;
  float ave_build_residual_time = 0.0;
  float ave_ekf_time = 0.0;
  int scan_count = 0;
  StatesGroup state_;
  V3D position_last_;

  V3D last_slide_position = {0,0,0};

  geometry_msgs::Quaternion geoQuat_;

  int feats_down_size_;
  int effct_feat_num_;
  std::vector<M3D> cross_mat_list_;
  std::vector<M3D> body_cov_list_;
  std::vector<pointWithVar> pv_list_;
  std::vector<PointToPlane> ptpl_list_;
  KD_TREE<PointType> fastlio_ikdtree_;
  std::vector<PointVector> fastlio_nearest_points_;
  std::vector<uint8_t> fastlio_neighbor_valid_;
  BoxPointType fastlio_local_map_points_{};
  bool fastlio_local_map_initialized_ = false;

  VoxelMapManager(VoxelMapConfig &config_setting, std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> &voxel_map)
      : config_setting_(config_setting), voxel_map_(voxel_map),
        lio_diagnostics_(diagnosticsConfig(config_setting), std::string(ROOT_DIR) + "Log/diagnostics",
                         [](const std::string &message) {
                           RCLCPP_WARN(rclcpp::get_logger("laserMapping"), "%s", message.c_str());
                         })
  {
    current_frame_id_ = 0;
    feats_undistort_.reset(new PointCloudXYZI());
    feats_down_body_.reset(new PointCloudXYZI());
    feats_down_world_.reset(new PointCloudXYZI());
  };

  void StateEstimation(StatesGroup &state_propagat, double scan_time = -1.0);
  M3D boundedStateCovarianceBlock(int start_index) const;
  void TransformLidar(const Eigen::Matrix3d rot, const Eigen::Vector3d t, const PointCloudXYZI::Ptr &input_cloud,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud);

  void BuildVoxelMap();
  // Only voxel-plane uses accumulation. ikd-tree keeps its single-scan seed.
  // Optional output is populated only on the transition to READY, in world
  // coordinates. Existing callers retain the same map/gate behavior.
  bool TryInitializeVoxelMap(double scan_time, std::vector<pointWithVar> *accepted_seed = nullptr);
  const VoxelMapInitializationStatus &initializationStatus() const { return initialization_status_; }
  V3F RGBFromVoxel(const V3D &input_point);

  void UpdateVoxelMap(const std::vector<pointWithVar> &input_points);
  void UpdateFastLioMap(const PointCloudXYZI::Ptr &world_cloud);
  bool FastLioMapReady() const;
  // Once per deskewed/downsampled scan; iteration poses are applied at matching time.
  void PrepareSourceNormals(bool geometry_requested = false);

  void BuildResidualListOMP(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list);
  void BuildResidualListFastLio(std::vector<pointWithVar> &pv_list, std::vector<PointToPlane> &ptpl_list,
                                bool recompute_neighbors, int iteration = -1);

  void build_single_residual(pointWithVar &pv, const VoxelOctoTree *current_octo, const int current_layer, bool &is_sucess, double &prob,
                             PointToPlane &single_ptpl, fast_livo2::diagnostics::GeometryPoint *record = nullptr,
                             const VOXEL_LOCATION &root_voxel = VOXEL_LOCATION());

  void pubVoxelMap();

  void mapSliding();
  void clearMemOutOfMap(const int& x_max,const int& x_min,const int& y_max,const int& y_min,const int& z_max,const int& z_min );

private:
  static fast_livo2::diagnostics::Config diagnosticsConfig(const VoxelMapConfig &config)
  {
    auto result = config.diagnostics_;
    result.correspondence_mode = config.correspondence_mode_;
    return result;
  }
  fast_livo2::diagnostics::Logger lio_diagnostics_;
  std::vector<fast_livo2::diagnostics::GeometryPoint> diagnostic_geometry_;
  std::vector<fast_livo2::normal_consistency::SourceNormal> source_normals_;
  fast_livo2::diagnostics::NormalStats normal_stats_;
  void captureGeometry(int iteration, const StatesGroup &prior,
                             const StatesGroup &linearization, const StatesGroup &after) noexcept;
  void logXicp(double scan_time, int iteration, const fast_livo2::xicp::Detection &detection,
               const fast_livo2::xicp::Update &update, const StatesGroup &propagated,
               const StatesGroup &before, double elapsed_ms);
  std::ofstream xicp_log_;
  bool xicp_log_attempted_ = false;
  std::vector<pointWithVar> initialMapPoints() const;
  void buildVoxelMapFromPoints(const std::vector<pointWithVar> &input_points);
  void resetVoxelInitialization();
  void reportVoxelInitialization(double scan_time);
  std::vector<pointWithVar> initialization_points_;
  VoxelMapInitializationStatus initialization_status_;
  double initialization_start_time_ = -1.0;
  double initialization_last_time_ = -1.0;
  double initialization_log_time_ = -1.0;
  void BuildFastLioMap();
  void SegmentFastLioLocalMap();
  bool estimateFastLioPlane(const PointVector &points, V3D &normal, double &d, V3D &center) const;
  void constrainGravityState(bool project_covariance);
  void GetUpdatePlane(const VoxelOctoTree *current_octo, const int pub_max_voxel_layer, std::vector<VoxelPlane> &plane_list);

  void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string plane_ns, const VoxelPlane &single_plane, const float alpha,
                      const Eigen::Vector3d rgb);
  void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec, geometry_msgs::Quaternion &q);

  void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b);
};
typedef std::shared_ptr<VoxelMapManager> VoxelMapManagerPtr;

#endif // VOXEL_MAP_H_
