#include "xicp.h"

#include <Eigen/Cholesky>
#include <stdexcept>

namespace fast_livo2::xicp
{
Update update(const Eigen::MatrixXd &prior_covariance, const Eigen::MatrixXd &jacobian,
              const Eigen::VectorXd &inverse_variance, const Eigen::VectorXd &residual,
              const Eigen::VectorXd &prior_delta, const Eigen::MatrixXd &constraints,
              const Eigen::MatrixXd &target_gain)
{
  const int size = prior_covariance.rows(), count = jacobian.rows(), rows = constraints.rows();
  if (size < 6 || prior_covariance.cols() != size || jacobian.cols() != 6 || count < 1 ||
      inverse_variance.size() != count || residual.size() != count || prior_delta.size() != size ||
      constraints.cols() != size || rows > 6 || target_gain.rows() != rows || target_gain.cols() != count ||
      !prior_covariance.allFinite() || !jacobian.allFinite() || !inverse_variance.allFinite() ||
      !residual.allFinite() || !prior_delta.allFinite() || !constraints.allFinite() || !target_gain.allFinite() ||
      (inverse_variance.array() <= 0.0).any())
    throw std::invalid_argument("Invalid X-ICP Kalman update input");

  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(size, size);
  const Eigen::MatrixXd covariance = 0.5 * (prior_covariance + prior_covariance.transpose());
  Eigen::LDLT<Eigen::MatrixXd> prior_solver(covariance);
  if (prior_solver.info() != Eigen::Success || !prior_solver.isPositive() ||
      (prior_solver.vectorD().array() <= 0.0).any())
    throw std::runtime_error("X-ICP prior covariance is not positive definite");
  const Eigen::MatrixXd weighted_transpose = jacobian.transpose() * inverse_variance.asDiagonal();
  Eigen::MatrixXd information = prior_solver.solve(identity);
  information.topLeftCorner(6, 6).noalias() += weighted_transpose * jacobian;
  information = (0.5 * (information + information.transpose())).eval();
  Eigen::LDLT<Eigen::MatrixXd> posterior_solver(information);
  if (posterior_solver.info() != Eigen::Success || !posterior_solver.isPositive() ||
      (posterior_solver.vectorD().array() <= 0.0).any())
    throw std::runtime_error("X-ICP posterior information solve failed");
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(size, count);
  rhs.topRows(6) = weighted_transpose;
  Update result;
  result.gain = posterior_solver.solve(rhs);
  if (rows > 0)
  {
    const Eigen::MatrixXd posterior_ct = posterior_solver.solve(constraints.transpose());
    Eigen::MatrixXd schur = constraints * posterior_ct;
    schur = (0.5 * (schur + schur.transpose())).eval();
    Eigen::LDLT<Eigen::MatrixXd> schur_solver(schur);
    if (schur_solver.info() != Eigen::Success || !schur_solver.isPositive() ||
        (schur_solver.vectorD().array() <= 0.0).any())
      throw std::runtime_error("X-ICP constraints are numerically dependent");
    // Exact KKT/Schur solution of the constrained quadratic objective.
    const Eigen::MatrixXd violation = constraints * result.gain - target_gain;
    result.gain -= posterior_ct * schur_solver.solve(violation);
    result.constraint_error = (constraints * result.gain - target_gain).norm();
    if (result.constraint_error > 1e-7 * (1.0 + target_gain.norm()))
      throw std::runtime_error("X-ICP equality constraint solve inaccurate");
  }
  const Eigen::VectorXd innovation = residual - jacobian * prior_delta.head(6);
  result.correction = prior_delta + result.gain * innovation;

  // Joseph covariance using the actual constrained gain. A NONE direction
  // suppresses a LiDAR correction; it is NOT an exact position observation.
  // D's dependence on the same residual is already included in this gain.
  Eigen::MatrixXd remaining = identity;
  remaining.leftCols(6).noalias() -= result.gain * jacobian;
  const Eigen::MatrixXd noise_gain = result.gain * inverse_variance.cwiseInverse().cwiseSqrt().asDiagonal();
  result.covariance = remaining * covariance * remaining.transpose() + noise_gain * noise_gain.transpose();
  result.covariance = (0.5 * (result.covariance + result.covariance.transpose())).eval();
  if (!result.correction.allFinite() || !result.covariance.allFinite() || !result.gain.allFinite())
    throw std::runtime_error("X-ICP produced a non-finite update");
  return result;
}
} // namespace fast_livo2::xicp
