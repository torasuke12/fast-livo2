#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>

#include <cmath>
#include <iostream>

// Exercise allocation inside the installed PCL library followed by destruction
// in this translation unit. On AVX hosts -march=native used to give the caller
// a different Eigen allocator ABI and crash on the first output destruction.
int main()
{
  std::cout << "Eigen alignment " << EIGEN_DEFAULT_ALIGN_BYTES
            << ", malloc already aligned " << EIGEN_MALLOC_ALREADY_ALIGNED << '\n';
  for (int trial = 1; trial <= 16; ++trial)
  {
    auto input = pcl::PointCloud<pcl::PointXYZINormal>::Ptr(
        new pcl::PointCloud<pcl::PointXYZINormal>);
    for (int i = 0; i < trial * 1000; ++i)
    {
      pcl::PointXYZINormal point{};
      point.x = (i % 101) * .03f;
      point.y = ((i / 101) % 99) * .03f;
      point.z = (i % 7) * .1f;
      point.normal_z = 1.f;
      input->push_back(point);
    }
    pcl::VoxelGrid<pcl::PointXYZINormal> filter;
    filter.setLeafSize(.07f, .07f, .07f);
    filter.setInputCloud(input);
    pcl::PointCloud<pcl::PointXYZINormal> output;
    filter.filter(output);
    if (output.empty() || output.size() >= input->size()) return 1;
    for (const auto& point : output)
      if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
          !std::isfinite(point.z)) return 2;
  }
  return 0;
}
