#include "voxel_map.h"
#include <pcl/filters/voxel_grid.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace
{
void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

VoxelMapConfig testConfig()
{
  VoxelMapConfig c{};
  c.correspondence_mode_ = "voxel_plane";
  c.max_voxel_size_ = 0.5;
  c.max_layer_ = 2;
  c.layer_init_num_ = {5, 5, 5, 5, 5};
  c.max_points_num_ = 100;
  c.planner_threshold_ = 0.0001;
  c.dept_err_ = 0.02;
  c.beam_err_ = 0.05;
  c.max_state_cov_for_matching_ = 0.01;
  c.fastlio_filter_size_map_ = 0.5;
  c.fastlio_cube_side_length_ = 1000.0;
  c.fastlio_detection_range_ = 100.0;
  c.initialization_.min_points = 500;
  c.initialization_.min_planes = 10;
  c.initialization_.max_points = 20000;
  return c;
}

using Points = std::vector<V3D>;

Points face(int axis, double offset = -2.25)
{
  Points points;
  for (int i = 0; i < 31; ++i)
    for (int j = 0; j < 31; ++j)
    {
      V3D p;
      p[axis] = offset;
      p[(axis + 1) % 3] = 1.03 + i * 0.08;
      p[(axis + 2) % 3] = 4.07 + j * 0.08;
      points.push_back(p);
    }
  return points;
}

Points threeFaces()
{
  Points points;
  for (int axis = 0; axis < 3; ++axis)
  {
    const auto patch = face(axis);
    points.insert(points.end(), patch.begin(), patch.end());
  }
  return points;
}

struct Fixture
{
  VoxelMapConfig config;
  std::unordered_map<VOXEL_LOCATION, VoxelOctoTree *> empty_map;
  // ikd-tree contains a large fixed-capacity rebuild queue. As in LIVMapper,
  // allocate its manager on the heap rather than the test thread's stack.
  std::unique_ptr<VoxelMapManager> storage;
  VoxelMapManager &manager;

  explicit Fixture(VoxelMapConfig c = testConfig())
      : config(c), storage(new VoxelMapManager(config, empty_map)), manager(*storage)
  {
    manager.extR_.setIdentity();
    manager.extT_.setZero();
  }

  ~Fixture()
  {
    for (auto &entry : manager.voxel_map_) delete entry.second;
    manager.voxel_map_.clear();
  }

  bool add(const Points &world_points, double time, std::vector<pointWithVar> *seed = nullptr)
  {
    manager.feats_down_body_->clear();
    manager.feats_down_world_->clear();
    for (const auto &w : world_points)
    {
      const V3D b = manager.state_.rot_end.transpose() * (w - manager.state_.pos_end);
      PointType pb{}, pw{};
      pb.x = b.x(); pb.y = b.y(); pb.z = b.z();
      pw.x = w.x(); pw.y = w.y(); pw.z = w.z();
      manager.feats_down_body_->push_back(pb);
      manager.feats_down_world_->push_back(pw);
    }
    manager.feats_down_size_ = world_points.size();
    return manager.TryInitializeVoxelMap(time, seed);
  }
};

void testAccumulationAndWorldFrame()
{
  auto config = testConfig();
  config.initialization_.min_points = 2500;
  Fixture f(config);
  const auto scene = threeFaces();
  size_t total = 0;
  bool ready = false;
  std::vector<pointWithVar> seed;
  Points expected_seed;
  for (size_t slice = 0; slice < 8; ++slice)
  {
    Points points;
    for (size_t i = slice; i < scene.size(); i += 8) points.push_back(scene[i]);
    // Every slice is expressed in a different body pose. The accumulator must
    // preserve the world coordinates provided with each individual slice.
    f.manager.state_.rot_end = Eigen::AngleAxisd(0.03 * slice, V3D::UnitY()).toRotationMatrix();
    f.manager.state_.pos_end = V3D(0.02 * slice, -0.01 * slice, 0.0);
    const StatesGroup before = f.manager.state_;
    ready = f.add(points, 10.0 + 0.04 * slice, &seed);
    expected_seed.insert(expected_seed.end(), points.begin(), points.end());
    if (!ready) require(seed.empty(), "Unready accumulation exported a seed");
    total += points.size();
    require(f.manager.initializationStatus().points == total, "Slices were discarded instead of accumulated");
    require((f.manager.state_ - before).norm() < 1e-12, "Initialization performed a state update");
    require((f.manager.state_.cov - before.cov).norm() < 1e-12, "Initialization changed covariance");
    if (total < 2500) require(!ready, "Sparse point gate opened too early");
    if (ready) break;
  }
  require(ready, "Distributed three-plane scene never initialized");
  require(seed.size() == expected_seed.size(), "Exported seed lost earlier scans");
  for (size_t i = 0; i < seed.size(); ++i)
    require((seed[i].point_w - expected_seed[i]).norm() < 1e-5,
            "Exported seed changed acquisition world coordinates");
  const auto status = f.manager.initializationStatus();
  require(status.scans > 1 && status.planes >= 10, "Initialization did not use multiple scans/planes");
  require(status.normal_eigenvalues.x() > 0.1, "Three normal directions were not recognized");
  require(std::abs(status.normal_eigenvalues.sum() - 1.0) < 1e-10, "Normal information was not normalized");
  for (const auto &entry : f.manager.voxel_map_)
  {
    const auto &plane = *entry.second->plane_ptr_;
    if (plane.is_plane_)
      require((plane.center_.array() + 2.25).abs().minCoeff() < 1e-5,
              "Accumulated points were re-transformed with a later body pose");
  }
  const size_t map_size = f.manager.voxel_map_.size();
  require(f.add(face(0, -20.0), 11.0, &seed), "Ready state was not latched");
  require(seed.empty(), "Latched READY re-exported or retained a stale seed");
  require(f.manager.voxel_map_.size() == map_size && f.manager.initializationStatus().points == status.points,
          "A ready seed was rebuilt");
}

void testDirectionAndTimeGates()
{
  Fixture f;
  require(!f.add(face(0), 20.0), "Minimum time gate ignored");
  require(!f.add(face(0, 2.25), 20.2), "Opposite parallel walls counted as independent directions");
  require(f.manager.initializationStatus().planes >= 10, "Test lacks sufficient planes");
  require(f.manager.initializationStatus().normal_eigenvalues.x() < 1e-8, "Parallel walls should be rank deficient");
  require(!f.add(face(1), 20.4), "Two normal directions were accepted as three");
  require(f.add(face(2), 20.6), "Third independent direction did not release gate");

  auto c = testConfig();
  c.initialization_.min_duration_sec = 0.5;
  Fixture duration(c);
  require(!duration.add(threeFaces(), 30.0), "Single complete scan bypassed time gate");
  require(!duration.add(threeFaces(), 30.1), "Time gate opened early");
  require(duration.add(threeFaces(), 30.6), "Time gate never opened");

  c = testConfig();
  c.initialization_.min_planes = 10000;
  Fixture plane_count(c);
  plane_count.add(threeFaces(), 35.0);
  require(!plane_count.add(threeFaces(), 35.2), "Plane-count gate ignored");
  require(plane_count.manager.initializationStatus().normal_eigenvalues.x() > 0.1,
          "Plane-count test did not otherwise have directional coverage");
}

void testBoundsAndInvalidData()
{
  Fixture timeout;
  timeout.add(face(0), 40.0);
  timeout.add(face(0), 40.2);
  require(!timeout.add(face(0), 42.1), "Timeout forced a degenerate initialization");
  require(timeout.manager.initializationStatus().resets == 1 && timeout.manager.initializationStatus().scans == 1,
          "Timeout did not reset the pending window");
  require(!timeout.add(face(0), 41.0), "Time reversal forced initialization");
  const auto before = timeout.manager.initializationStatus();
  require(before.resets == 2, "Time reversal did not reset the window");
  timeout.add(face(0), 41.0);
  require(timeout.manager.initializationStatus().scans == before.scans, "Duplicate timestamp was counted twice");

  auto c = testConfig();
  c.initialization_.max_points = 1500;
  Fixture capacity(c);
  capacity.add(face(0), 50.0);
  require(!capacity.add(face(0), 50.2), "Point limit forced initialization");
  require(capacity.manager.initializationStatus().points <= 1500 && capacity.manager.initializationStatus().resets == 1,
          "Point limit did not bound the buffer");
  capacity.add(threeFaces(), 50.4);
  require(capacity.manager.initializationStatus().points == 1500, "Oversized scan exceeded capacity");

  Fixture invalid;
  require(!invalid.add({}, 60.0), "Empty scan initialized");
  require(!invalid.add({V3D(std::numeric_limits<double>::quiet_NaN(), 1.0, 1.0)}, 60.1), "NaN scan initialized");
  require(!invalid.add(face(0), std::numeric_limits<double>::quiet_NaN()), "Invalid timestamp initialized");
  require(invalid.manager.initializationStatus().points == 0, "Invalid points entered accumulation");

  c.initialization_.min_duration_sec = 0.0;
  c.initialization_.min_points = 6;
  c.initialization_.min_planes = 3;
  Fixture coincident(c);
  require(!coincident.add(Points(20, V3D(1.1, 2.1, 3.1)), 65.0), "Coincident points formed a usable plane");
  require(coincident.manager.initializationStatus().planes == 0, "Coincident points counted as a plane");
  Points line;
  for (int i = 0; i < 20; ++i) line.push_back(V3D(1.01 + 0.01 * i, 2.1, 3.1));
  Fixture collinear(c);
  require(!collinear.add(line, 65.0), "Collinear points formed a usable plane");
  require(collinear.manager.initializationStatus().planes == 0, "Collinear points counted as a plane");
}

void testIkdTreeBypassAndConfiguration()
{
  auto c = testConfig();
  c.correspondence_mode_ = "fastlio_ikdtree";
  c.initialization_.min_points = 100000;
  c.initialization_.min_planes = 10000;
  Fixture ikd(c);
  require(!ikd.add(face(0), 70.0), "voxel accumulator ran in ikd-tree mode");
  ikd.manager.BuildVoxelMap();
  require(ikd.manager.FastLioMapReady(), "ikd-tree no longer initialized from one scan");
  require(ikd.manager.initializationStatus().points == 0, "ikd-tree populated voxel initialization buffer");

  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("lio.correspondence_mode", "voxel_plane"),
                               rclcpp::Parameter("lio.voxel_init_min_normal_eigenvalue", 0.5)});
  auto node = std::make_shared<rclcpp::Node>("voxel_init_invalid_config_test", options);
  VoxelMapConfig bad;
  bool rejected = false;
  try { loadVoxelConfig(node, bad); }
  catch (const std::invalid_argument &) { rejected = true; }
  require(rejected, "Impossible normal-eigenvalue threshold was not rejected");
  options.parameter_overrides({rclcpp::Parameter("lio.correspondence_mode", "fastlio_ikdtree"),
                               rclcpp::Parameter("lio.voxel_init_min_normal_eigenvalue", 0.5)});
  auto ikd_node = std::make_shared<rclcpp::Node>("ikd_unused_voxel_config_test", options);
  VoxelMapConfig unused;
  loadVoxelConfig(ikd_node, unused); // Unused voxel thresholds must not block ikd-tree.
}

void testBinaryPclOutputLifetime()
{
  // Exercise the actual binary PCL filter followed by destruction in our
  // native-architecture build. An Eigen allocator mismatch faults on release.
  pcl::VoxelGrid<PointType> filter;
  filter.setLeafSize(0.5f, 0.5f, 0.5f);
  auto input = PointCloudXYZI::Ptr(new PointCloudXYZI());
  auto output = PointCloudXYZI::Ptr(new PointCloudXYZI());
  for (int pass = 1; pass <= 3; ++pass)
  {
    input->clear();
    for (int cell = 0; cell < pass * 100; ++cell)
      for (float offset : {0.1f, 0.2f})
      {
        PointType p{};
        p.x = cell + offset; p.y = 0.1f; p.z = 0.1f;
        input->push_back(p);
      }
    filter.setInputCloud(input);
    output->reserve(input->size());
    filter.filter(*output);
    require(output->size() == static_cast<size_t>(pass * 100), "PCL downsample lost voxel centroids");
    for (size_t i = 0; i < output->size(); ++i)
      require(std::abs((*output)[i].x - (i + 0.15)) < 1e-4, "PCL downsample changed centroid coordinates");
  }
}
} // namespace

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  int result = 0;
  try
  {
    testAccumulationAndWorldFrame();
    testDirectionAndTimeGates();
    testBoundsAndInvalidData();
    testIkdTreeBypassAndConfiguration();
    testBinaryPclOutputLifetime();
    std::cout << "voxel initialization regression tests passed" << std::endl;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    result = 1;
  }
  rclcpp::shutdown();
  return result;
}
