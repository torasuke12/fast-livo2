#include "lio_diagnostics.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <locale>
#include <stdexcept>

namespace fast_livo2::diagnostics
{
namespace
{
constexpr int kStateSize = 19;
using Clock = std::chrono::steady_clock;

void number(std::ostream &out, double value)
{
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value,
                                  std::chars_format::general, std::numeric_limits<double>::max_digits10);
  if (result.ec != std::errc()) throw std::runtime_error("number formatting failed");
  out.write(buffer, result.ptr - buffer);
}

void field(std::ostream &out, double value) { out << ','; number(out, value); }

void header(std::ostream &out, const char *name, int rows, int columns)
{
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < columns; ++j) out << ',' << name << '_' << i << '_' << j;
}

template <typename Derived>
void matrix(std::ostream &out, const Eigen::MatrixBase<Derived> &value)
{
  for (int i = 0; i < value.rows(); ++i)
    for (int j = 0; j < value.cols(); ++j) field(out, value(i, j));
}

std::string quote(const std::string &text)
{
  std::string result = "\"";
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : text)
  {
    if (c == '"' || c == '\\') { result += '\\'; result += c; }
    else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
    else result += c;
  }
  return result + '"';
}

const char *statusName(ScanStatus status)
{
  switch (status)
  {
  case ScanStatus::Updated: return "updated";
  case ScanStatus::NoMatches: return "no_matches";
  case ScanStatus::Rejected: return "rejected";
  case ScanStatus::NoMap: return "no_map";
  default: return "no_update";
  }
}

void openFile(std::ofstream &file, const std::filesystem::path &path)
{
  file.exceptions(std::ios::failbit | std::ios::badbit);
  file.imbue(std::locale::classic());
  file.open(path);
}
} // namespace

void Config::validate() const
{
  if (!std::isfinite(detail_start_sec) || !std::isfinite(detail_end_sec) ||
      detail_start_sec < 0.0 || detail_end_sec <= detail_start_sec)
    throw std::invalid_argument("lio.diagnostics.detail requires finite 0 <= start_sec < end_sec");
  if (geometry_enabled && !detail_enabled)
    throw std::invalid_argument("lio.diagnostics.geometry.enabled requires detail.enabled");
}

Logger::Logger(const Config &config, const std::string &default_directory,
               std::function<void(const std::string &)> warning)
    : config_(config), default_directory_(default_directory), warning_(std::move(warning))
{
  config_.validate();
}

void Logger::fail(const std::string &reason) noexcept
{
  if (failed_) return;
  failed_ = true;
  scan_open_ = false;
  try
  {
    const std::string message = "LIO diagnostics disabled: " + reason;
    if (warning_) warning_(message);
    else std::cerr << message << '\n';
  }
  catch (...) {} // A diagnostic failure must not change filter control flow.
}

void Logger::open()
{
  namespace fs = std::filesystem;
  const fs::path base(config_.output_dir.empty() ? default_directory_ : config_.output_dir);
  fs::create_directories(base);
  const auto stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  // create_directory is atomic: independent nodes cannot overwrite one another.
  for (unsigned suffix = 0;; ++suffix)
  {
    const auto path = base / ("run-" + std::to_string(stamp) + "-" + std::to_string(suffix));
    if (fs::create_directory(path)) { run_directory_ = path.string(); break; }
  }
  std::ofstream metadata;
  openFile(metadata, fs::path(run_directory_) / "metadata.json");
  const bool voxel = config_.correspondence_mode == "voxel_plane";
  metadata << "{\n  \"schema_version\": " << (voxel ? 4 : 3)
           << ",\n  \"correspondence_mode\": " << quote(config_.correspondence_mode)
           << ",\n  \"matrix_order\": \"row-major\",\n"
           << "  \"state_order\": [\"rot_x\",\"rot_y\",\"rot_z\",\"pos_x\",\"pos_y\",\"pos_z\","
              "\"inverse_exposure\",\"vel_x\",\"vel_y\",\"vel_z\",\"bg_x\",\"bg_y\",\"bg_z\","
              "\"ba_x\",\"ba_y\",\"ba_z\",\"gravity_x\",\"gravity_y\",\"gravity_z\"],\n"
           << "  \"frames\": \"rotation: right IMU tangent; position/velocity/gravity: world; biases: IMU\",\n"
           << "  \"units\": \"rotation rad, position m, velocity m/s, bg rad/s, ba/gravity m/s^2, inverse_exposure native unscaled scalar\",\n"
           << "  \"time\": \"seconds from first LiDAR; same scan_time as X-ICP CSV\",\n"
           << "  \"residual\": \"negative signed point-to-plane distance (m)\",\n"
           << "  \"innovation\": \"residual - H * prior_delta.head(6)\",\n"
           << "  \"covariances\": \"P_prior: propagated tangent; P_iter: iteration tangent; P_after: returned state tangent after transport/projection\",\n"
           << "  \"gain\": \"pre-transport gain; detail rows 7..15 are velocity, gyro bias, accel bias\",\n"
           << "  \"iteration_logging_ms\": \"iteration statistics and CSV writes only; excludes gain extraction and scan-row write/flush\",\n"
           << "  \"xicp_enabled\": " << (config_.xicp_enabled ? "true" : "false") << ",\n"
           << "  \"detail_enabled\": " << (config_.detail_enabled ? "true" : "false") << ",\n"
           << "  \"detail_final_only\": " << (config_.detail_final_only ? "true" : "false") << ",\n"
           << "  \"geometry_enabled\": " << (config_.geometry_enabled ? "true" : "false") << ",\n"
           << "  \"geometry\": \"last attempted iteration per scan in detail window; updated scans use the committed iteration\",\n"
           << "  \"geometry_point_index\": \"source_index is the deskewed/downsampled LiDAR point index within this scan; match_index joins correspondences.point_index, -1 if rejected\",\n"
           << "  \"geometry_map\": " << quote(voxel ?
                "selected plane; for rejection, furthest matching stage then smallest distance/radius or normalized residual; ties keep first visited candidate; not all candidates or a full map snapshot" :
                "actual cached nearest-neighbor coordinates for every query, including rejected points; not a complete tree snapshot or persistent map IDs") << ",\n"
           << "  \"geometry_status\": \"0 invalid point, 1 insufficient neighbors, 2 neighbor distance, 3 plane fit, 4 residual, 5 accepted, 6 normal mismatch, 7 no voxel, 8 no plane, 9 outside extent, 10 invalid variance; voxel plane_valid marks available plane fields\",\n"
           << "  \"geometry_frames\": \"lidar: deskewed LiDAR; world: float-rounded matching query; search_world: query at last neighbor search; normals/centers/neighbors: world; extrinsic maps LiDAR to IMU\",\n"
           << "  \"geometry_padding\": \"unavailable numeric fields are zero (plane_id/layer -1); invalid input/variance may be non-finite; no 5-NN or source normals in voxel-plane schema\",\n"
           << "  \"voxel_variances\": \"gate uses linearization query and point_world_covariance; update uses prior query and rotated point_body_covariance plus 0.001 m^2; update fields valid only when match_index>=0\",\n"
           << "  \"voxel_extent\": \"tangential_distance <= 3*radius; gate_distance is float(abs(signed distance)); gate passes strictly below sigma_num*sqrt(gate_variance)\",\n"
           << "  \"voxel_covariance_order\": \"plane_covariance: world normal then world center; point_body_covariance: LiDAR; point_world_covariance: matching world\",\n"
           << "  \"detail_start_sec\": "; number(metadata, config_.detail_start_sec);
  metadata << ",\n  \"detail_end_sec\": "; number(metadata, config_.detail_end_sec);
  metadata << ",\n  \"source_quality\": \"0 not computed, 1 usable, 2 invalid query, 3 insufficient neighbors, 4 radius, 5 degenerate, 6 nonplanar, 7 narrow, 8 numerical failure\",\n"
           << "  \"normal_comparisons\": \"last attempted iteration, after original residual gate; angle invalid is nan; source normals in LiDAR frame\",\n"
           << "  \"normal_stats\": \"source valid/invalid counts and PCA time once per scan; comparison/rejection counts at last attempted iteration\",\n"
           << "  \"ros_parameters\": {";
  bool first = true;
  for (const auto &entry : config_.parameters)
  {
    if (!first) metadata << ',';
    metadata << '\n' << "    " << quote(entry.first) << ": " << quote(entry.second);
    first = false;
  }
  metadata << "\n  }\n}\n";
  metadata.close();

  openFile(iterations_file_, fs::path(run_directory_) / "iterations.csv");
  iterations_file_ << "scan_id,time,iteration,final,matches,xicp_enabled,constraints,gain_constraint_error,"
                     "distance_mean,residual_mean,residual_rms,residual_max_abs,innovation_mean,innovation_rms,innovation_max_abs,innovation_wrss";
  header(iterations_file_, "HWH", 6, 6);
  header(iterations_file_, "HTWinnovation", 6, 1);
  header(iterations_file_, "P_iter", kStateSize, kStateSize);
  header(iterations_file_, "KH", kStateSize, 6);
  for (const char *name : {"gain_norm", "gain_noise_diag", "Kinnovation", "prior_delta", "solution"})
    header(iterations_file_, name, kStateSize, 1);
  header(iterations_file_, "C", 6, kStateSize);
  header(iterations_file_, "target", 6, 1);
  iterations_file_ << '\n';

  openFile(scans_file_, fs::path(run_directory_) / "scans.csv");
  scans_file_ << "scan_id,time,status,iterations,last_iteration,committed_iteration,iteration_logging_ms";
  header(scans_file_, "P_prior", kStateSize, kStateSize);
  header(scans_file_, "P_after", kStateSize, kStateSize);
  scans_file_ << ",geometry_logging_ms,source_normal_ms,source_normal_valid,source_normal_invalid,normal_comparisons,normal_rejected\n";
  if (config_.detail_enabled)
  {
    openFile(details_file_, fs::path(run_directory_) / "correspondences.csv");
    details_file_ << "scan_id,time,iteration,point_index,distance,residual,innovation,inverse_variance";
    header(details_file_, "H", 1, 6);
    for (int row = 7; row <= 15; ++row) details_file_ << ",K_" << row;
    details_file_ << '\n';
  }
  if (config_.geometry_enabled)
  {
    openFile(geometry_file_, fs::path(run_directory_) / "geometry.csv");
    if (voxel)
    {
      geometry_file_ << "scan_id,time,iteration,source_index,match_index,status,plane_valid,plane_id,layer,support_points,update_enabled,root_voxel_x,root_voxel_y,root_voxel_z";
      for (const char *name : {"lidar", "world", "normal", "center", "voxel_center", "plane_eigenvalues"}) header(geometry_file_, name, 3, 1);
      geometry_file_ << ",plane_d,distance,radius,tangential_distance,gate_distance,gate_plane_variance,gate_point_variance,gate_variance,probability,update_plane_variance,update_point_variance,update_variance";
      header(geometry_file_, "plane_covariance", 6, 6);
      header(geometry_file_, "point_world_covariance", 3, 3);
      header(geometry_file_, "point_body_covariance", 3, 3);
      geometry_file_ << '\n';
    }
    else
    {
    geometry_file_ << "scan_id,time,iteration,source_index,match_index,status,search_iteration,search_farthest_squared,neighbor_count";
    for (const char *name : {"lidar", "world", "search_world", "normal", "center"}) header(geometry_file_, name, 3, 1);
    geometry_file_ << ",plane_d,distance,plane_fit_rms,plane_fit_max";
    header(geometry_file_, "neighbors", 5, 3);
    header(geometry_file_, "source_normal", 3, 1);
    header(geometry_file_, "source_eigenvalues", 3, 1);
    geometry_file_ << ",source_usable,source_quality,source_neighbor_count,source_max_neighbor_distance,normal_angle_valid,normal_angle_deg\n";
    }
    openFile(geometry_states_file_, fs::path(run_directory_) / "geometry_states.csv");
    geometry_states_file_ << "scan_id,time,iteration,points";
    for (const std::string prefix : {"prior", "linearization", "after"})
    {
      header(geometry_states_file_, (prefix + "_R").c_str(), 3, 3);
      for (const std::string name : {"position", "velocity", "bg", "ba", "gravity"})
        header(geometry_states_file_, (prefix + "_" + name).c_str(), 3, 1);
      geometry_states_file_ << ',' << prefix << "_inverse_exposure";
    }
    header(geometry_states_file_, "extrinsic_R", 3, 3);
    header(geometry_states_file_, "extrinsic_t", 3, 1);
    geometry_states_file_ << '\n';
  }
}

void Logger::beginScan(double time, const Eigen::Ref<const Eigen::MatrixXd> &prior) noexcept
{
  if (!enabled()) return;
  try
  {
    if (run_directory_.empty()) open();
    if (prior.rows() != kStateSize || prior.cols() != kStateSize) throw std::invalid_argument("prior must be 19x19");
    prior_ = prior;
    time_ = time;
    scan_id_ = next_scan_id_++;
    iterations_ = 0;
    committed_iteration_ = -2;
    iteration_logging_ms_ = 0.0;
    geometry_logging_ms_ = 0.0;
    normal_stats_ = {};
    scan_open_ = true;
  }
  catch (const std::exception &error) { fail(error.what()); }
  catch (...) { fail("unknown open/begin failure"); }
}

void Logger::iteration(int index, bool final, const Eigen::MatrixXd &H,
                       const Eigen::VectorXd &inverse_variance, const Eigen::VectorXd &residual,
                       const Eigen::Ref<const Eigen::VectorXd> &prior_delta,
                       const Eigen::Ref<const Eigen::VectorXd> &solution,
                       const Eigen::Ref<const Eigen::MatrixXd> &P_iter, const Eigen::MatrixXd &gain,
                       const Eigen::MatrixXd &constraints, const Eigen::MatrixXd &target_gain,
                       double gain_constraint_error) noexcept
{
  if (!enabled() || !scan_open_) return;
  const auto started = Clock::now();
  try
  {
    const int count = H.rows(), rows = constraints.rows();
    if (count < 1 || H.cols() != 6 || inverse_variance.size() != count || residual.size() != count ||
        prior_delta.size() != kStateSize || solution.size() != kStateSize ||
        P_iter.rows() != kStateSize || P_iter.cols() != kStateSize ||
        gain.rows() != kStateSize || gain.cols() != count || rows > 6 ||
        constraints.cols() != kStateSize || target_gain.rows() != rows || target_gain.cols() != count ||
        (inverse_variance.array() <= 0.0).any()) throw std::invalid_argument("invalid iteration dimensions/weights");
    const Eigen::VectorXd innovation = residual - H * prior_delta.head(6);
    const Eigen::MatrixXd weighted_transpose = H.transpose() * inverse_variance.asDiagonal();
    const Eigen::MatrixXd information = weighted_transpose * H;
    const Eigen::VectorXd score = weighted_transpose * innovation;
    const Eigen::MatrixXd KH = gain * H;
    const Eigen::VectorXd gain_norm = gain.rowwise().norm();
    const Eigen::VectorXd noise = gain.array().square().matrix() * inverse_variance.cwiseInverse();
    const Eigen::VectorXd contribution = gain * innovation;
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(6, kStateSize);
    Eigen::VectorXd target = Eigen::VectorXd::Zero(6);
    if (rows) { C.topRows(rows) = constraints; target.head(rows) = target_gain * innovation; }
    auto &out = iterations_file_;
    out << scan_id_; field(out, time_);
    out << ',' << index << ',' << final << ',' << count << ',' << config_.xicp_enabled << ',' << rows;
    field(out, gain_constraint_error);
    field(out, -residual.mean()); field(out, residual.mean());
    field(out, residual.norm() / std::sqrt(count)); field(out, residual.cwiseAbs().maxCoeff());
    field(out, innovation.mean()); field(out, innovation.norm() / std::sqrt(count));
    field(out, innovation.cwiseAbs().maxCoeff());
    field(out, innovation.dot(inverse_variance.asDiagonal() * innovation));
    matrix(out, information); matrix(out, score); matrix(out, P_iter); matrix(out, KH);
    matrix(out, gain_norm); matrix(out, noise); matrix(out, contribution);
    matrix(out, prior_delta); matrix(out, solution); matrix(out, C); matrix(out, target);
    out << '\n';
    if (config_.detail_enabled && (!config_.detail_final_only || final) &&
        time_ >= config_.detail_start_sec && time_ < config_.detail_end_sec)
    {
      for (int i = 0; i < count; ++i)
      {
        details_file_ << scan_id_; field(details_file_, time_);
        details_file_ << ',' << index << ',' << i;
        field(details_file_, -residual(i)); field(details_file_, residual(i));
        field(details_file_, innovation(i)); field(details_file_, inverse_variance(i));
        matrix(details_file_, H.row(i));
        for (int row = 7; row <= 15; ++row) field(details_file_, gain(row, i));
        details_file_ << '\n';
      }
    }
    ++iterations_;
    if (final) committed_iteration_ = index;
    iteration_logging_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  }
  catch (const std::exception &error) { fail(error.what()); }
  catch (...) { fail("unknown iteration failure"); }
}

bool Logger::geometryEnabled() const
{
  return enabled() && scan_open_ && config_.geometry_enabled &&
         time_ >= config_.detail_start_sec && time_ < config_.detail_end_sec;
}

void Logger::geometry(int index, const GeometryState &prior, const GeometryState &linearization,
                      const GeometryState &after, const Eigen::Matrix3d &extrinsic_R,
                      const Eigen::Vector3d &extrinsic_t, const std::vector<GeometryPoint> &points) noexcept
{
  if (!geometryEnabled()) return;
  const auto started = Clock::now();
  try
  {
    auto &frame = geometry_states_file_;
    frame << scan_id_; field(frame, time_); frame << ',' << index << ',' << points.size();
    for (const auto *state : {&prior, &linearization, &after})
    {
      matrix(frame, state->rotation); matrix(frame, state->position); matrix(frame, state->velocity);
      matrix(frame, state->bg); matrix(frame, state->ba); matrix(frame, state->gravity);
      field(frame, state->inverse_exposure);
    }
    matrix(frame, extrinsic_R); matrix(frame, extrinsic_t); frame << '\n';
    for (size_t source = 0; source < points.size(); ++source)
    {
      const auto &point = points[source];
      if (config_.correspondence_mode == "voxel_plane")
      {
        const auto &v = point.voxel;
        auto &out = geometry_file_;
        out << scan_id_; field(out, time_);
        out << ',' << index << ',' << source << ',' << point.match_index << ',' << static_cast<int>(point.status)
            << ',' << v.plane_valid << ',' << v.plane_id << ',' << v.layer << ',' << v.support_points << ',' << v.update_enabled;
        for (auto key : v.root_voxel) out << ',' << key;
        for (const auto *value : {&point.lidar, &point.world, &point.normal, &point.center, &v.voxel_center, &v.eigenvalues}) matrix(out, *value);
        for (double value : {point.plane_d, point.distance, v.radius, v.tangential_distance, v.gate_distance,
                             v.gate_plane_variance, v.gate_point_variance, v.gate_variance, v.probability,
                             v.update_plane_variance, v.update_point_variance, v.update_variance}) field(out, value);
        matrix(out, v.plane_covariance); matrix(out, v.point_world_covariance); matrix(out, v.point_body_covariance);
        out << '\n';
        continue;
      }
      if (point.neighbor_count < 0 || point.neighbor_count > 5)
        throw std::invalid_argument("invalid geometry neighbor count");
      auto &out = geometry_file_;
      out << scan_id_; field(out, time_);
      out << ',' << index << ',' << source << ',' << point.match_index << ',' << static_cast<int>(point.status)
          << ',' << point.search_iteration;
      field(out, point.search_farthest_squared); out << ',' << point.neighbor_count;
      matrix(out, point.lidar); matrix(out, point.world); matrix(out, point.search_world);
      matrix(out, point.normal); matrix(out, point.center);
      field(out, point.plane_d); field(out, point.distance);
      double squared_sum = 0.0, max_error = 0.0;
      if (point.status == MatchStatus::Residual || point.status == MatchStatus::Accepted || point.status == MatchStatus::NormalMismatch)
        for (int j = 0; j < point.neighbor_count; ++j)
        {
          const double error = point.normal.dot(point.neighbors[j]) + point.plane_d;
          squared_sum += error * error;
          max_error = std::max(max_error, std::abs(error));
        }
      field(out, point.neighbor_count ? std::sqrt(squared_sum / point.neighbor_count) : 0.0);
      field(out, max_error);
      for (int j = 0; j < 5; ++j)
      {
        if (j < point.neighbor_count) matrix(out, point.neighbors[j]);
        else { field(out, 0.0); field(out, 0.0); field(out, 0.0); }
      }
      matrix(out, point.source.normal); matrix(out, point.source.eigenvalues);
      out << ',' << point.source.usable() << ',' << static_cast<int>(point.source.quality)
          << ',' << point.source.neighbor_count;
      field(out, point.source.max_neighbor_distance);
      out << ',' << point.comparison.valid;
      field(out, point.comparison.valid ? point.comparison.angle_deg : std::numeric_limits<double>::quiet_NaN());
      out << '\n';
    }
    geometry_logging_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  }
  catch (const std::exception &error) { fail(error.what()); }
  catch (...) { fail("unknown geometry capture failure"); }
}

void Logger::endScan(ScanStatus status, int last_iteration,
                     const Eigen::Ref<const Eigen::MatrixXd> &after) noexcept
{
  if (!enabled() || !scan_open_) return;
  try
  {
    if (after.rows() != kStateSize || after.cols() != kStateSize) throw std::invalid_argument("after must be 19x19");
    scans_file_ << scan_id_; field(scans_file_, time_);
    scans_file_ << ',' << statusName(status) << ',' << iterations_ << ',' << last_iteration << ',' << committed_iteration_;
    field(scans_file_, iteration_logging_ms_);
    matrix(scans_file_, prior_); matrix(scans_file_, after);
    field(scans_file_, geometry_logging_ms_);
    field(scans_file_, normal_stats_.computation_ms);
    scans_file_ << ',' << normal_stats_.valid << ',' << normal_stats_.invalid
                << ',' << normal_stats_.comparisons << ',' << normal_stats_.rejected << '\n';
    iterations_file_.flush();
    if (config_.detail_enabled) details_file_.flush();
    if (config_.geometry_enabled) { geometry_file_.flush(); geometry_states_file_.flush(); }
    scans_file_.flush();
    scan_open_ = false;
  }
  catch (const std::exception &error) { fail(error.what()); }
  catch (...) { fail("unknown end/flush failure"); }
}
} // namespace fast_livo2::diagnostics
