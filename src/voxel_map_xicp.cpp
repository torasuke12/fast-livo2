#include "voxel_map.h"

#include <iomanip>
#include <filesystem>

void VoxelMapManager::logXicp(const double scan_time, const int iteration,
                             const fast_livo2::xicp::Detection &detection,
                             const fast_livo2::xicp::Update &update, const StatesGroup &propagated,
                             const StatesGroup &before, const double elapsed_ms)
{
  if (!xicp_log_attempted_)
  {
    xicp_log_attempted_ = true;
    const auto directory = config_setting_.debug_log_dir_.empty() ?
        std::filesystem::path(ROOT_DIR) / "Log" : std::filesystem::path(config_setting_.debug_log_dir_);
    const std::string path = (directory / "xicp_diagnostic.csv").string();
    xicp_log_.open(path, std::ios::out | std::ios::trunc);
    if (!xicp_log_)
    {
      RCLCPP_WARN(rclcpp::get_logger("laserMapping"), "Cannot write X-ICP diagnostics: %s", path.c_str());
      return;
    }
    xicp_log_ << "time,iteration,matches,constraints,gain_constraint_error,solve_ms";
    for (const std::string label : {"prior", "before", "after"})
    {
      xicp_log_ << ',' << label << "_vx," << label << "_vy," << label << "_vz," << label << "_forward_speed," << label << "_pitch";
      for (const std::string quantity : {"bg", "ba"})
        for (const char axis : {'x', 'y', 'z'}) xicp_log_ << ',' << label << '_' << quantity << '_' << axis;
    }
    for (int i = 0; i < 6; ++i)
    {
      const std::string prefix = std::string(i < 3 ? "r" : "t") + std::to_string(i % 3);
      for (const std::string field : {"category", "eigenvalue", "combined", "strong", "selected", "rank_fallback",
                                      "axis_lx", "axis_ly", "axis_lz", "axis_wx", "axis_wy", "axis_wz"})
        xicp_log_ << ',' << prefix << '_' << field;
    }
    xicp_log_ << '\n';
    xicp_log_ << std::setprecision(12);
  }
  if (!xicp_log_) return;
  xicp_log_ << scan_time << ',' << iteration << ',' << effct_feat_num_ << ',' << detection.constraints.rows()
             << ',' << update.constraint_error << ',' << elapsed_ms;
  for (const StatesGroup *state : std::array<const StatesGroup *, 3>{&propagated, &before, &state_})
  {
    const double forward = (state->rot_end.transpose() * state->vel_end).x();
    const double pitch = std::atan2(-state->rot_end(2, 0), std::hypot(state->rot_end(0, 0), state->rot_end(1, 0)));
    xicp_log_ << ',' << state->vel_end.x() << ',' << state->vel_end.y() << ',' << state->vel_end.z()
              << ',' << forward << ',' << pitch;
    for (const V3D *bias : {&state->bias_g, &state->bias_a})
      xicp_log_ << ',' << bias->x() << ',' << bias->y() << ',' << bias->z();
  }
  for (const auto &direction : detection.directions)
  {
    const V3D world_axis = before.rot_end * extR_ * direction.axis;
    xicp_log_ << ',' << static_cast<int>(direction.category) << ',' << direction.eigenvalue
              << ',' << direction.combined << ',' << direction.strong << ',' << direction.selected
              << ',' << direction.rank_fallback;
    for (int i = 0; i < 3; ++i) xicp_log_ << ',' << direction.axis(i);
    for (int i = 0; i < 3; ++i) xicp_log_ << ',' << world_axis(i);
  }
  xicp_log_ << '\n';
  // Keep the last completed scan even if the legacy node crashes on SIGINT.
  xicp_log_.flush();
}
