#include "lio_diagnostics.h"
#include "xicp.h"

#include <Eigen/Dense>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using fast_livo2::diagnostics::Config;
using fast_livo2::diagnostics::Logger;
using fast_livo2::diagnostics::ScanStatus;
using Row = std::map<std::string, std::string>;

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

std::vector<std::string> split(const std::string &line)
{
  std::vector<std::string> fields;
  std::istringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

std::vector<Row> csv(const fs::path &path)
{
  std::ifstream file(path);
  require(bool(file), "missing CSV " + path.string());
  std::string line;
  std::getline(file, line);
  const auto names = split(line);
  std::vector<Row> rows;
  while (std::getline(file, line))
  {
    const auto values = split(line);
    require(values.size() == names.size(), "CSV column mismatch");
    Row row;
    for (size_t i = 0; i < names.size(); ++i) row.emplace(names[i], values[i]);
    rows.push_back(std::move(row));
  }
  return rows;
}

double value(const Row &row, const std::string &key) { return std::stod(row.at(key)); }

void close(double actual, double expected, const std::string &message)
{
  require(std::abs(actual - expected) <= 1e-11 * (1.0 + std::abs(expected)), message);
}

void checkMatrix(const Row &row, const std::string &name, const Eigen::MatrixXd &expected)
{
  for (int i = 0; i < expected.rows(); ++i)
    for (int j = 0; j < expected.cols(); ++j)
      close(value(row, name + '_' + std::to_string(i) + '_' + std::to_string(j)), expected(i, j), name);
}

void numericTest(const fs::path &root, bool constrained)
{
  Config config;
  config.enabled = true;
  config.detail_enabled = true;
  config.output_dir = root.string();
  config.xicp_enabled = constrained;
  config.parameters["test.quoted"] = "quote\" newline\n";
  Logger logger(config, "unused");
  Eigen::MatrixXd B(19, 19), H(8, 6);
  for (int i = 0; i < B.size(); ++i) B.data()[i] = std::sin(0.1 * i) * 0.1;
  for (int i = 0; i < H.size(); ++i) H.data()[i] = std::cos(0.7 * i);
  const Eigen::MatrixXd P = B * B.transpose() + Eigen::MatrixXd::Identity(19, 19) * 0.1;
  Eigen::VectorXd W = Eigen::VectorXd::LinSpaced(8, 10.0, 80.0);
  Eigen::VectorXd residual = Eigen::VectorXd::LinSpaced(8, -0.07, 0.12);
  Eigen::VectorXd delta = Eigen::VectorXd::LinSpaced(19, -0.01, 0.01);
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(constrained ? 1 : 0, 19);
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(C.rows(), 8);
  if (constrained) { C(0, 3) = 1.0; D(0, 0) = 0.05; }
  const auto update = fast_livo2::xicp::update(P, H, W, residual, delta, C, D);
  const auto P_saved = P; const auto H_saved = H; const auto gain_saved = update.gain;
  Eigen::MatrixXd transport = Eigen::MatrixXd::Identity(19, 19);
  transport.topLeftCorner(3, 3) = Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::MatrixXd after = transport * update.covariance * transport.transpose();

  // Includes the lower boundary, excludes the upper; early convergence at -1.
  for (double time : {169.999, 170.0, 219.999, 220.0})
  {
    logger.beginScan(time, P * 2.0);
    logger.iteration(-1, true, H, W, residual, delta, update.correction, P, update.gain, C, D, update.constraint_error);
    logger.endScan(ScanStatus::Updated, -1, after);
  }
  require(logger.enabled(), "valid diagnostic was disabled");
  require((P - P_saved).norm() == 0 && (H - H_saved).norm() == 0 && (update.gain - gain_saved).norm() == 0,
          "observer mutated input");
  const fs::path run(logger.runDirectory());
  const auto iterations = csv(run / "iterations.csv");
  const auto scans = csv(run / "scans.csv");
  const auto points = csv(run / "correspondences.csv");
  require(iterations.size() == 4 && scans.size() == 4 && points.size() == 16, "wrong record count or detail boundary");
  checkMatrix(iterations[1], "P_iter", P);
  checkMatrix(scans[1], "P_prior", P * 2.0);
  checkMatrix(scans[1], "P_after", after);
  require(scans[1].at("status") == "updated" && value(scans[1], "committed_iteration") == -1,
          "early convergence not marked committed");

  Eigen::MatrixXd saved_H(8, 6), saved_K(9, 8);
  Eigen::VectorXd saved_W(8), saved_residual(8), saved_innovation(8);
  for (int i = 0; i < 8; ++i)
  {
    const Row &row = points[i];
    require(value(row, "time") == 170.0 && value(row, "point_index") == i, "detail identity");
    for (int j = 0; j < 6; ++j) saved_H(i, j) = value(row, "H_0_" + std::to_string(j));
    for (int j = 0; j < 9; ++j) saved_K(j, i) = value(row, "K_" + std::to_string(j + 7));
    saved_W(i) = value(row, "inverse_variance");
    saved_residual(i) = value(row, "residual"); saved_innovation(i) = value(row, "innovation");
    close(value(row, "distance"), -saved_residual(i), "distance sign");
  }
  require((saved_innovation - (saved_residual - saved_H * delta.head(6))).norm() < 1e-12, "innovation definition");
  checkMatrix(iterations[1], "HWH", saved_H.transpose() * saved_W.asDiagonal() * saved_H);
  checkMatrix(iterations[1], "HTWinnovation", saved_H.transpose() * saved_W.asDiagonal() * saved_innovation);
  const Eigen::MatrixXd saved_KH = saved_K * saved_H;
  const Eigen::VectorXd contribution = saved_K * saved_innovation;
  const Eigen::VectorXd noise = saved_K.array().square().matrix() * saved_W.cwiseInverse();
  for (int i = 0; i < 9; ++i)
  {
    const std::string row = std::to_string(i + 7);
    close(value(iterations[1], "Kinnovation_" + row + "_0"), contribution(i), "gain contribution");
    close(update.correction(i + 7) - delta(i + 7), contribution(i), "actual correction");
    close(value(iterations[1], "gain_norm_" + row + "_0"), saved_K.row(i).norm(), "gain norm");
    close(value(iterations[1], "gain_noise_diag_" + row + "_0"), noise(i), "gain noise");
    for (int j = 0; j < 6; ++j) close(value(iterations[1], "KH_" + row + '_' + std::to_string(j)), saved_KH(i, j), "KH");
  }
  if (constrained) checkMatrix(iterations[1], "target", (Eigen::VectorXd(6) << (D * saved_innovation)(0), 0, 0, 0, 0, 0).finished());

  // Uncommitted iterations remain visible but are not an accepted scan update.
  for (ScanStatus status : {ScanStatus::Rejected, ScanStatus::NoMatches})
  {
    logger.beginScan(221, P);
    logger.iteration(-1, false, H, W, residual, delta, update.correction, P, update.gain, C, D, update.constraint_error);
    logger.endScan(status, 0, P);
  }
  logger.beginScan(222, P); logger.endScan(ScanStatus::NoMap, -2, P);
  const auto results = csv(run / "scans.csv");
  require(results[4].at("status") == "rejected" && value(results[4], "committed_iteration") == -2, "rejection status");
  require(results[5].at("status") == "no_matches" && value(results[5], "committed_iteration") == -2, "empty rematch status");
  require(results[6].at("status") == "no_map" && value(results[6], "iterations") == 0, "map status");
  checkMatrix(results[4], "P_after", P);
  // A second logger sharing a parent directory must never overwrite this run.
  Logger second(config, "unused"); second.beginScan(0, P); second.endScan(ScanStatus::NoUpdate, -2, P);
  require(second.runDirectory() != logger.runDirectory(), "run directory collision");
}

void geometryTest(const fs::path &root)
{
  using namespace fast_livo2::diagnostics;
  Config config;
  config.enabled = true; config.detail_enabled = true; config.detail_final_only = true;
  config.geometry_enabled = true; config.output_dir = root.string();
  Logger logger(config, "unused");
  Eigen::MatrixXd P = Eigen::MatrixXd::Identity(19, 19);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 6), K = Eigen::MatrixXd::Zero(19, 1);
  H(0, 5) = 1; K(7, 0) = 0.1;
  Eigen::VectorXd W = Eigen::VectorXd::Constant(1, 1000), residual = Eigen::VectorXd::Constant(1, -0.03);
  Eigen::VectorXd delta = Eigen::VectorXd::Zero(19), solution = K * residual;
  Eigen::MatrixXd C(0, 19), D(0, 1);
  GeometryState prior, before, after;
  prior.position = Eigen::Vector3d(1.2345678901234567, 2, 3);
  before.rotation = Eigen::AngleAxisd(0.12345, Eigen::Vector3d::UnitY()).toRotationMatrix();
  after.velocity = Eigen::Vector3d(.5, .1, .2);
  std::vector<GeometryPoint> points(4);
  points[0].status = MatchStatus::Accepted; points[0].match_index = 0;
  points[0].normal = Eigen::Vector3d::UnitZ(); points[0].plane_d = -2;
  points[0].world = Eigen::Vector3d(1, 1, 2.03); points[0].distance = .03;
  points[0].neighbor_count = 5; points[0].search_iteration = -1;
  points[0].search_world = Eigen::Vector3d(1, 1, 2.01);
  points[0].search_farthest_squared = .5;
  for (int j = 0; j < 5; ++j) points[0].neighbors[j] = Eigen::Vector3d(j * .1, 1, 2);
  points[1] = points[0]; points[1].status = MatchStatus::Residual; points[1].match_index = -1;
  points[2].status = MatchStatus::InsufficientNeighbors; points[2].neighbor_count = 2;
  points[2].neighbors[0] = Eigen::Vector3d(1, 2, 3); points[2].neighbors[1] = Eigen::Vector3d(2, 3, 4);
  points[3] = points[0]; points[3].status = MatchStatus::NormalMismatch; points[3].match_index = -1;
  points[3].source.quality = fast_livo2::normal_consistency::Quality::Usable;
  points[3].source.normal = Eigen::Vector3d::UnitX(); points[3].source.neighbor_count = 20;
  points[3].source.eigenvalues = Eigen::Vector3d(.0001, .02, .03);
  points[3].source.max_neighbor_distance = .4;
  points[3].comparison.valid = true; points[3].comparison.angle_deg = 79;
  for (double time : {169.999, 170.0, 219.999, 220.0})
  {
    logger.beginScan(time, P);
    require(logger.geometryEnabled() == (time >= 170 && time < 220), "geometry boundary");
    logger.iteration(-1, false, H, W, residual, delta, solution, P, K, C, D, 0);
    logger.iteration(0, true, H, W, residual, delta, solution, P, K, C, D, 0);
    logger.geometry(0, prior, before, after, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), points);
    logger.endScan(ScanStatus::Updated, 0, P);
    require(!logger.geometryEnabled(), "geometry capture outside scan");
  }
  // A terminal failed attempt has geometry but no committed point/gain rows.
  logger.beginScan(190, P);
  logger.iteration(-1, false, H, W, residual, delta, solution, P, K, C, D, 0);
  logger.geometry(0, prior, before, prior, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero(), points);
  logger.endScan(ScanStatus::Rejected, 0, P);
  require(logger.enabled(), "geometry logger disabled");
  const fs::path run(logger.runDirectory());
  const auto detail = csv(run / "correspondences.csv"), geometry = csv(run / "geometry.csv");
  const auto states = csv(run / "geometry_states.csv"), scans = csv(run / "scans.csv");
  require(detail.size() == 2 && geometry.size() == 12 && states.size() == 3, "geometry/final-only counts");
  require(value(detail[0], "iteration") == 0, "non-final detail leaked");
  require(value(geometry[0], "search_iteration") == -1 && value(geometry[0], "iteration") == 0, "cached search identity");
  require(value(geometry[1], "match_index") == -1 && value(geometry[1], "status") == 4, "rejected point identity");
  close(value(geometry[0], "plane_fit_max"), 0, "plane fit residual");
  close(value(geometry[3], "status"), 6, "normal rejection reason preserved");
  close(value(geometry[3], "normal_angle_deg"), 79, "normal angle roundtrip");
  close(value(geometry[3], "source_eigenvalues_1_0"), .02, "source PCA roundtrip");
  require(std::isnan(value(geometry[0], "normal_angle_deg")), "unavailable angle must be nan");
  close(value(geometry[2], "neighbors_1_2"), 4, "neighbor coordinates");
  close(value(geometry[2], "neighbors_2_2"), 0, "unused neighbor padding");
  checkMatrix(states[0], "prior_position", prior.position);
  checkMatrix(states[0], "linearization_R", before.rotation);
  checkMatrix(states[0], "after_velocity", after.velocity);
  require(scans.back().at("status") == "rejected" && value(scans.back(), "committed_iteration") == -2, "failed geometry commit status");
}

int main()
{
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path root = fs::temp_directory_path() / ("lio-diagnostics-test-" + std::to_string(stamp));
  try
  {
    Config disabled; disabled.output_dir = (root / "disabled").string();
    Logger off(disabled, "unused");
    const Eigen::MatrixXd P = Eigen::MatrixXd::Identity(19, 19);
    off.beginScan(0, P); off.endScan(ScanStatus::Updated, -1, P);
    require(!fs::exists(root), "disabled logger wrote files");
    Config invalid; invalid.detail_end_sec = invalid.detail_start_sec;
    bool threw = false;
    try { invalid.validate(); } catch (const std::invalid_argument &) { threw = true; }
    require(threw, "invalid interval accepted");
    invalid.detail_end_sec = std::numeric_limits<double>::quiet_NaN(); threw = false;
    try { invalid.validate(); } catch (const std::invalid_argument &) { threw = true; }
    require(threw, "non-finite interval accepted");
    numericTest(root / "baseline", false);
    numericTest(root / "xicp", true);
    geometryTest(root / "geometry");
    invalid = Config{}; invalid.geometry_enabled = true; threw = false;
    try { invalid.validate(); } catch (const std::invalid_argument &) { threw = true; }
    require(threw, "geometry without detail accepted");
    const auto blocked = root / "not-a-directory";
    std::ofstream(blocked) << "test";
    Config failure; failure.enabled = true; failure.output_dir = blocked.string();
    int warnings = 0;
    Logger bad(failure, "unused", [&](const std::string &) { ++warnings; });
    bad.beginScan(0, P); bad.beginScan(1, P); bad.endScan(ScanStatus::Updated, -1, P);
    require(!bad.enabled() && warnings == 1, "I/O failure must disable diagnostics and warn once");
    fs::remove_all(root);
    std::cout << "LIO diagnostics: round trips, constrained/unconstrained gains, covariance stages, boundaries, statuses, disabled and I/O failure passed\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << " (artifacts: " << root << ")\n";
    return 1;
  }
}
