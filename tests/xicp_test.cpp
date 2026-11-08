#include "xicp.h"

#include <Eigen/Dense>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace Eigen;
using namespace fast_livo2::xicp;

namespace
{
void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}
void near(const MatrixXd &a, const MatrixXd &b, double tolerance, const char *message)
{
  require(a.rows() == b.rows() && a.cols() == b.cols() && a.allFinite() && b.allFinite() &&
              (a - b).norm() <= tolerance * (1.0 + b.norm()), message);
}
void throws(const std::function<void()> &function)
{
  try { function(); }
  catch (const std::exception &) { return; }
  throw std::runtime_error("Expected invalid input to be rejected");
}
Matrix3d rotation(double angle, const Vector3d &axis)
{
  return AngleAxisd(angle, axis.normalized()).toRotationMatrix();
}
MatrixXd randomNormals(int count)
{
  MatrixXd normals = MatrixXd::Random(count, 3);
  for (int i = 0; i < count; ++i) normals.row(i).normalize();
  return normals;
}

void thresholdsAndZeroMoment()
{
  for (int count : {0, 34, 35, 179, 180, 250})
  {
    MatrixXd points = MatrixXd::Zero(count, 3), normals = MatrixXd::Zero(count, 3);
    normals.col(2).setOnes();
    const Detection result = detect(points, normals, Config{});
    for (int i = 0; i < 5; ++i) require(result.directions[i].category == Category::None, "Unobserved axis not NONE");
    const Category expected = count < 35 ? Category::None : count < 180 ? Category::Partial : Category::Full;
    require(result.directions[5].category == expected, "Paper threshold boundary incorrect");
    require(std::abs(result.directions[5].combined - count) < 1e-10, "Contribution must be a sum");
    if (expected == Category::Partial)
      near(result.target_gain.row(5), MatrixXd::Constant(1, count, 1.0 / count), 1e-12, "Partial translation LS target incorrect");
  }
}

void rotationCappingAndSigns()
{
  MatrixXd points(80, 3), normals(80, 3);
  points.rowwise() = Vector3d(100, 0, 0).transpose();
  normals.rowwise() = Vector3d(0, 1, 0).transpose();
  const Detection result = detect(points, normals, Config{});
  require(result.directions[2].category == Category::Partial, "Long lever arms inflated information");
  require(std::abs(result.directions[2].combined - 80) < 1e-10, "Rotation moment not capped");
  VectorXd residual = VectorXd::LinSpaced(80, -0.1, 0.2);
  MatrixXd flipped = normals;
  VectorXd flipped_residual = residual;
  for (int i = 0; i < 80; i += 2) { flipped.row(i) *= -1; flipped_residual(i) *= -1; }
  const Detection other = detect(points, flipped, Config{});
  near(other.constraints, result.constraints, 1e-12, "Normal sign changed constraints");
  near(other.target_gain * flipped_residual, result.target_gain * residual, 1e-12, "Normal sign changed targets");
  points *= 0.001; // moment 0.1 must stay 0.1 rather than become 1
  require(detect(points, normals, Config{}).directions[2].category == Category::None, "Small rotation moments amplified");
}

void combinedInformationBranch()
{
  for (int count : {320, 440})
  {
    MatrixXd points = MatrixXd::Zero(count, 3), normals(count, 3);
    for (int i = 0; i < count; ++i)
      normals.row(i) << (i & 1 ? -0.6 : 0.6), (i & 2 ? -0.5 : 0.5),
                        (i & 4 ? -std::sqrt(0.39) : std::sqrt(0.39));
    const auto result = detect(points, normals, Config{});
    for (int axis = 3; axis < 6; ++axis)
    {
      const auto &direction = result.directions[axis];
      require(direction.strong == 0, "Synthetic combined-only case has strong contributions");
      const Category expected = direction.combined >= 250 ? Category::Full :
                                direction.combined >= 180 ? Category::Partial : Category::None;
      require(direction.category == expected, "Combined-information decision branch incorrect");
      if (expected == Category::Partial)
        require(direction.selected == count, "Combined branch must sample weak and strong information");
    }
    require(result.directions[4].category == (count == 320 ? Category::Partial : Category::Full),
             "k1 combined-information boundary not exercised");
  }
}

void partialThreeDimensionalSolve()
{
  MatrixXd points = 4 * MatrixXd::Random(90, 3), normals = randomNormals(90);
  Config config;
  config.enough_information = 1000; config.insufficient_information = 900; config.minimal_information = 1;
  const Detection result = detect(points, normals, config);
  const VectorXd residual = VectorXd::Random(90);
  int row = 0, partial = 0;
  for (int axis = 0; axis < 6; ++axis)
  {
    const auto &direction = result.directions[axis];
    if (direction.category == Category::Full) continue;
    if (direction.category == Category::Partial)
    {
      ++partial;
      Matrix3d h = Matrix3d::Zero(); Vector3d b = Vector3d::Zero();
      for (int i = 0; i < normals.rows(); ++i)
      {
        Vector3d g = normals.row(i);
        if (axis < 3) { const Vector3d p = points.row(i); g = p.cross(g); }
        const Vector3d capped = axis < 3 ? g / std::max(1.0, g.norm()) : g;
        if (std::abs(capped.dot(direction.axis)) < config.contribution_strong) continue;
        h += g * g.transpose(); b += g * residual(i);
      }
      const double expected = direction.axis.dot(h.completeOrthogonalDecomposition().solve(b));
      require(std::abs(result.target_gain.row(row).dot(residual) - expected) < 1e-9, "Not a 3D auxiliary solve");
    }
    ++row;
  }
  require(partial == 6, "Synthetic partial case did not exercise all six auxiliary solves");
}

void rankFallback()
{
  MatrixXd points = MatrixXd::Zero(50, 3), normals(50, 3);
  normals.topRows(40).rowwise() = Vector3d(1, 0, 0).transpose();
  normals.bottomRows(10).rowwise() = Vector3d(0.2, std::sqrt(0.96), 0).transpose();
  const auto result = detect(points, normals, Config{});
  bool fallback = false;
  for (const auto &direction : result.directions) fallback |= direction.rank_fallback;
  require(fallback && result.target_gain.isZero(), "Rank-deficient subset must not invent an unconstrained component");
}

void localFrameInvariance()
{
  MatrixXd points = 3 * MatrixXd::Random(350, 3), normals = randomNormals(350);
  const auto a = detect(points, normals, Config{});
  const Matrix3d q = rotation(1.4, Vector3d(1, 2, 3));
  const auto b = detect(points * q.transpose(), normals * q.transpose(), Config{});
  for (int i = 0; i < 6; ++i)
  {
    require(a.directions[i].category == b.directions[i].category, "Frame change changed classification");
    require(std::abs(a.directions[i].combined - b.directions[i].combined) < 1e-8, "Frame change changed contribution");
    require(std::abs(std::abs(b.directions[i].axis.dot(q * a.directions[i].axis)) - 1) < 1e-10, "Frame change not propagated to axes");
  }
}

void leverArmJacobian()
{
  const Matrix3d r = rotation(0.7, Vector3d(1, -2, 3)), e = rotation(-0.3, Vector3d(2, 1, 4));
  const Vector3d t(0.3, -0.1, 0.07), p(2, 1, -0.4), n = Vector3d(0.5, 0.8, 0.2).normalized();
  const MatrixXd mapping = localPoseJacobian(r, e, t, 19);
  Matrix<double, 1, 6> local;
  local << p.cross((r * e).transpose() * n).transpose(), ((r * e).transpose() * n).transpose();
  const MatrixXd actual = local * mapping;
  MatrixXd numerical = MatrixXd::Zero(1, 19);
  const Vector3d body_point = e * p + t;
  for (int i = 0; i < 6; ++i)
  {
    const Vector3d basis = Vector3d::Unit(i % 3);
    if (i < 3)
      numerical(i) = n.dot(r * (rotation(1e-6, basis) - rotation(-1e-6, basis)) * body_point) / 2e-6;
    else numerical(i) = n(i - 3);
  }
  near(actual, numerical, 1e-9, "IMU/LiDAR lever-arm or rotation convention incorrect");
}

void constrainedFilter()
{
  constexpr int size = 19, count = 100;
  const MatrixXd random = MatrixXd::Random(size, size);
  const MatrixXd p = random * random.transpose() + MatrixXd::Identity(size, size) * 0.1;
  const MatrixXd j = MatrixXd::Random(count, 6);
  const VectorXd w = VectorXd::LinSpaced(count, 300, 1000), residual = VectorXd::Random(count), delta = VectorXd::Random(size);
  const auto full = update(p, j, w, residual, delta, MatrixXd(0, size), MatrixXd(0, count));
  MatrixXd h = MatrixXd::Zero(count, size); h.leftCols(6) = j;
  const MatrixXd a = p.inverse() + h.transpose() * w.asDiagonal() * h;
  const VectorXd b = h.transpose() * w.asDiagonal() * (residual - j * delta.head(6));
  near(full.correction, delta + a.ldlt().solve(b), 1e-9, "FULL not equivalent to original IEKF update");
  near(full.covariance, a.inverse(), 1e-9, "FULL covariance incorrect");

  MatrixXd c = MatrixXd::Zero(6, size); c.leftCols(6).setIdentity();
  auto none = update(p, j, w, residual, delta, c, MatrixXd::Zero(6, count));
  near(none.correction, delta, 1e-9, "All NONE updated pose, velocity, or bias");
  near(none.covariance, p, 1e-9, "All NONE shrank covariance");

  c = MatrixXd::Random(3, 6) * localPoseJacobian(rotation(0.6, Vector3d(1, 2, 3)), Matrix3d::Identity(), Vector3d(0.4, 0.2, 0.1), size);
  MatrixXd d = MatrixXd::Random(3, count) * 0.001;
  d.row(0).setZero(); // NONE; the others exercise PARTIAL data-dependent targets
  const auto constrained = update(p, j, w, residual, delta, c, d);
  MatrixXd kkt = MatrixXd::Zero(size + 3, size + 3);
  kkt.topLeftCorner(size, size) = a;
  kkt.topRightCorner(size, 3) = c.transpose(); kkt.bottomLeftCorner(3, size) = c;
  VectorXd rhs(size + 3); rhs.head(size) = b; rhs.tail(3) = d * (residual - j * delta.head(6));
  const VectorXd reference = kkt.fullPivLu().solve(rhs);
  near(constrained.correction, delta + reference.head(size), 1e-9, "Constrained update is not the KKT solution");
  near(c * (constrained.correction - delta), rhs.tail(3), 1e-9, "Constraint targets violated");
  near(c.row(0) * constrained.covariance * c.row(0).transpose(), c.row(0) * p * c.row(0).transpose(), 1e-9, "NONE falsely reduced directional covariance");
  SelfAdjointEigenSolver<MatrixXd> eig(constrained.covariance);
  require(eig.eigenvalues().minCoeff() > -1e-10, "Constrained covariance not PSD");
  // Independent linear error propagation, including correlation between
  // the auxiliary target and the original residual (shared measurement).
  MatrixXd error_mapping(size, size + count);
  error_mapping.leftCols(size) = MatrixXd::Identity(size, size) - constrained.gain * h;
  error_mapping.rightCols(count) = constrained.gain;
  MatrixXd input_cov = MatrixXd::Zero(size + count, size + count);
  input_cov.topLeftCorner(size, size) = p;
  input_cov.bottomRightCorner(count, count) = w.cwiseInverse().asDiagonal();
  near(constrained.covariance, error_mapping * input_cov * error_mapping.transpose(), 1e-10, "Shared-noise covariance inconsistent");

  MatrixXd bad = p; bad(0, 0) = -1e6;
  throws([&] { update(bad, j, w, residual, delta, c, d); });
  bad = p; bad(0, 0) = std::numeric_limits<double>::quiet_NaN();
  throws([&] { update(bad, j, w, residual, delta, c, d); });
  MatrixXd dependent = c; dependent.row(1) = dependent.row(0);
  throws([&] { update(p, j, w, residual, delta, dependent, d); });
}

void validation()
{
  Config config; config.validate(); config.minimal_information = 200;
  throws([&] { config.validate(); });
  config = Config{}; config.contribution_min = std::numeric_limits<double>::quiet_NaN();
  throws([&] { config.validate(); });
  MatrixXd points = MatrixXd::Zero(1, 3), normals = MatrixXd::Zero(1, 3);
  throws([&] { detect(points, normals, Config{}); });
  normals(0, 2) = 1; points(0, 0) = std::numeric_limits<double>::infinity();
  throws([&] { detect(points, normals, Config{}); });
}
} // namespace

int main()
{
  std::srand(47);
  try
  {
    thresholdsAndZeroMoment(); rotationCappingAndSigns(); combinedInformationBranch(); partialThreeDimensionalSolve(); rankFallback();
    localFrameInvariance(); leverArmJacobian(); constrainedFilter(); validation();
    std::cout << "X-ICP: 9 test groups passed\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "X-ICP test failed: " << error.what() << '\n';
    return 1;
  }
}
