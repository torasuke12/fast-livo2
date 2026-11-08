#include "LIVMapper.h"
#include <rosbag2_cpp/reader.hpp>
#include <rclcpp/serialization.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>
#include <optional>

template<class Message>
std::shared_ptr<Message> decode(const rosbag2_storage::SerializedBagMessage& record) {
  auto message=std::make_shared<Message>();
  rclcpp::SerializedMessage serialized(*record.serialized_data);
  rclcpp::Serialization<Message>().deserialize_message(&serialized,message.get());
  return message;
}

int main(int argc,char** argv) {
  if(argc<3) {
    std::cerr<<"Usage: fastlivo2_offline BAG NEW_OUTPUT [--ros-args --params-file FILE ...]\n";
    return 2;
  }
  const std::filesystem::path output=argv[2];
  if(std::filesystem::exists(output)) {
    std::cerr<<"Output already exists: "<<output<<'\n';return 2;
  }
  std::filesystem::create_directories(output/"scans");
  // Numeric Docker users may have no passwd home directory. Keep ROS logging
  // with this reconstruction instead of falling back to an unwritable /.ros.
  const auto ros_logs=std::filesystem::absolute(output/"ros").string();
  setenv("ROS_LOG_DIR",ros_logs.c_str(),0);
  rclcpp::init(argc,argv);
  int status=0;
  try {
    rclcpp::NodeOptions options;
    options.allow_undeclared_parameters(true).automatically_declare_parameters_from_overrides(true);
    options.append_parameter_override("debug.log_dir",(output/"logs").string());
    auto node=std::make_shared<rclcpp::Node>("laserMapping",options);
    cv::setNumThreads(1);Eigen::setNbThreads(1);
    LIVMapper mapper(node);
    mapper.initializeSubscribersAndPublishers();
    const double interval=fast_livo2::declareOrGet<double>(node,"offline.keyframe_interval",0.5);
    const double duration=fast_livo2::declareOrGet<double>(node,"offline.duration",0.0);
    const double rate=fast_livo2::declareOrGet<double>(node,"offline.rate",0.0);
    if(interval<=0 || duration<0 || !std::isfinite(rate) || rate<0)
      throw std::invalid_argument("Invalid offline interval/duration/rate");
    rosbag2_cpp::Reader reader;
    reader.open(argv[1]);
    const auto topics=reader.get_all_topics_and_types();
    const auto require_topic=[&](const std::string& name,const std::string& type) {
      const auto it=std::find_if(topics.begin(),topics.end(),[&](const auto& topic){return topic.name==name;});
      if(it==topics.end() || it->type!=type)
        throw std::runtime_error("Offline reconstruction requires "+name+" with type "+type);
    };
    require_topic(mapper.lid_topic,"sensor_msgs/msg/PointCloud2");
    require_topic(mapper.imu_topic,"sensor_msgs/msg/Imu");
    if(mapper.img_en) require_topic(mapper.img_topic,"sensor_msgs/msg/Image");
    rosbag2_storage::StorageFilter filter;
    filter.topics={mapper.lid_topic,mapper.imu_topic};
    if(mapper.img_en) filter.topics.push_back(mapper.img_topic);
    reader.set_filter(filter);
    std::ofstream poses(output/"poses.csv");
    poses<<"id,time,stamp,px,py,pz,qx,qy,qz,qw,vx,vy,vz,bgx,bgy,bgz,bax,bay,baz,points\n"<<std::setprecision(17);
    size_t imu_count=0,lidar_count=0,image_count=0,keyframes=0,updates=0;
    double last_saved=-1e20,last_time=0,last_report=-1;
    bool duration_reached=false;
    std::vector<Eigen::Vector3f> pending_cloud;
    auto drain=[&]() {
      while(mapper.sync_packages(mapper.LidarMeasures)) {
        mapper.handleFirstFrame();mapper.processImu();mapper.stateEstimationAndMapping();
        if(!mapper.lidar_map_inited) continue;
        // LIVO exports the posterior after image fusion, LIO after its scan update.
        if(mapper.LidarMeasures.lio_vio_flg!=VIO) {
          pending_cloud.clear();
          for(size_t i=0;i<mapper.feats_undistort->size();i+=2) {
            const auto& p=mapper.feats_undistort->points[i];
            const Eigen::Vector3d body=mapper.extR*Eigen::Vector3d(p.x,p.y,p.z)+mapper.extT;
            if(body.allFinite()) pending_cloud.push_back(body.cast<float>());
          }
        }
        if(mapper.img_en && mapper.LidarMeasures.lio_vio_flg!=VIO) continue;
        if(mapper.temporal_window_) {
          pending_cloud.clear();
          const auto& points=mapper.temporal_window_->nodes().back().points;
          for(size_t i=0;i<points.size();i+=2) pending_cloud.push_back(points[i].cast<float>());
        }
        const double stamp=mapper.LidarMeasures.last_lio_update_time;
        last_time=stamp-mapper._first_lidar_time;++updates;
        if(!mapper._state.pos_end.allFinite() || !mapper._state.rot_end.allFinite())
          throw std::runtime_error("Nonfinite reconstruction state");
        if(stamp-last_saved>=interval-1e-4 && !pending_cloud.empty()) {
          std::ofstream cloud(output/"scans"/(std::to_string(keyframes)+".bin"),std::ios::binary);
          size_t count=0;
          for(const auto& point:pending_cloud) {
            cloud.write(reinterpret_cast<const char*>(point.data()),3*sizeof(float));++count;
          }
          if(!cloud) throw std::runtime_error("Cannot write reconstruction cloud");
          // A flushed pose row commits this scan for the live RViz reader.
          cloud.close();
          if(!cloud) throw std::runtime_error("Cannot close reconstruction cloud");
          const auto& s=mapper._state;const Eigen::Quaterniond q(s.rot_end);
          poses<<keyframes<<','<<last_time<<','<<stamp;
          for(double v:{s.pos_end.x(),s.pos_end.y(),s.pos_end.z(),q.x(),q.y(),q.z(),q.w(),
                        s.vel_end.x(),s.vel_end.y(),s.vel_end.z(),s.bias_g.x(),s.bias_g.y(),s.bias_g.z(),
                        s.bias_a.x(),s.bias_a.y(),s.bias_a.z()}) poses<<','<<v;
          poses<<','<<count<<std::endl;
          if(!poses) throw std::runtime_error("Cannot write reconstruction poses");
          ++keyframes;last_saved=stamp;
        }
        if(last_time-last_report>=10) {
          std::cerr<<"OFFLINE t="<<last_time<<" keyframes="<<keyframes<<" position="
                   <<mapper._state.pos_end.transpose()<<std::endl;last_report=last_time;
        }
        if(duration>0 && last_time>=duration) {duration_reached=true;break;}
      }
    };
    std::optional<int64_t> first_record_time;
    const auto wall_start=std::chrono::steady_clock::now();
    while(reader.has_next() && rclcpp::ok() && !duration_reached) {
      const auto record=reader.read_next();
      if(!first_record_time) first_record_time=record->time_stamp;
      if(rate>0) {
        const auto target=wall_start+std::chrono::duration<double>(
            (record->time_stamp-*first_record_time)*1e-9/rate);
        while(rclcpp::ok() && std::chrono::steady_clock::now()<target)
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        if(!rclcpp::ok()) break;
      }
      if(record->topic_name==mapper.imu_topic) {mapper.imu_cbk(decode<sensor_msgs::msg::Imu>(*record));++imu_count;}
      else if(record->topic_name==mapper.lid_topic) {mapper.standard_pcl_cbk(decode<sensor_msgs::msg::PointCloud2>(*record));++lidar_count;}
      else if(record->topic_name==mapper.img_topic) {mapper.img_cbk(decode<sensor_msgs::msg::Image>(*record));++image_count;}
      drain();
    }
    if(!duration_reached) drain();
    if(keyframes<2 || !rclcpp::ok()) throw std::runtime_error("Reconstruction incomplete/interrupted");
    std::ofstream summary(output/"summary.json");
    summary<<std::setprecision(17)<<"{\"complete\":true,\"bag_exhausted\":"<<(!reader.has_next()?"true":"false")
      <<",\"imu_messages\":"<<imu_count<<",\"lidar_messages\":"<<lidar_count<<",\"image_messages\":"<<image_count
      <<",\"updates\":"<<updates<<",\"keyframes\":"<<keyframes<<",\"last_time\":"<<last_time<<"}\n";
  } catch(const std::exception& error) {
    std::ofstream(output/"failure.txt")<<error.what()<<'\n';
    std::cerr<<error.what()<<'\n';status=1;
  }
  rclcpp::shutdown();return status;
}
