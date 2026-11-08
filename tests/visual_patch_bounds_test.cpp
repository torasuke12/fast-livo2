#include "vio.h"
#include "visual_patch_bounds.h"
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

static void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  try {
    using fast_livo2::visualPatchInBounds;
    // The previous fixed 80-pixel border accepts this point, but level 3 plus
    // search level 2 reads 160 pixels away, before the image allocation.
    require(!visualPatchInBounds(640,480,90,90,8,5,true), "combined pyramid/search footprint accepted");
    require(visualPatchInBounds(640,480,320,240,8,5,true), "valid coarse footprint rejected");
    require(!visualPatchInBounds(640,480,std::numeric_limits<double>::quiet_NaN(),200,8,0,true), "NaN pixel accepted");
    require(!visualPatchInBounds(640,480,100,200,8,-1,true), "negative level accepted");
    // Enumerate the actual bilinear and central-difference accesses, including
    // float rounding at cell boundaries, independently of the bounding box.
    for (int level=0;level<=5;++level) for (bool gradient:{false,true})
      for (double u:{-1.,0.,3.99999999,40.,79.99999999,90.,159.99999999,160.,320.,479.99999999,600.,639.,640.})
        for (double v:{-1.,0.,40.,79.99999999,90.,160.,240.,319.99999999,439.99999999,479.,480.}) {
          const int scale=1<<level;
          const int x0=static_cast<int>(std::floor(static_cast<float>(u/scale)))*scale-4*scale;
          const int y0=static_cast<int>(std::floor(static_cast<float>(v/scale)))*scale-4*scale;
          bool valid=true;
          for (int x=0;x<8;++x) for (int y=0;y<8;++y)
            for (int dx=(gradient ? -1 : 0);dx<=(gradient ? 2 : 1);++dx)
              for (int dy=(gradient ? -1 : 0);dy<=(gradient ? 2 : 1);++dy) {
                const int px=x0+(x+dx)*scale,py=y0+(y+dy)*scale;
                valid=valid && px>=0 && px<640 && py>=0 && py<480;
              }
          require(visualPatchInBounds(640,480,u,v,8,level,gradient)==valid, "pixel footprint bounds differ from actual reads");
        }

    vk::PinholeCamera camera(640,480,600,600,320,240);
    StatesGroup state,prior;
    state.inv_expo_time=prior.inv_expo_time=1.;
    VIOManager manager;
    manager.cam=&camera;manager.state=&state;manager.state_propagat=&prior;
    manager.Rci=M3D::Identity();manager.Pci=V3D::Zero();
    manager.Jdphi_dR=M3D::Identity();manager.Jdp_dR=M3D::Zero();
    manager.fx=manager.fy=600;manager.cx=320;manager.cy=240;
    manager.width=640;manager.height=480;manager.patch_size=8;manager.patch_size_half=4;
    manager.patch_size_total=64;manager.max_iterations=2;manager.total_points=2;
    manager.img_point_cov=1000.;manager.exposure_estimate_en=false;
    manager.temporal_commit=[](){return true;};
    manager.visual_submap=new SubSparseMap();
    auto unsafe=std::make_unique<VisualPoint>(V3D((90.-320.)/600.*5.,(90.-240.)/600.*5.,5.));
    auto safe=std::make_unique<VisualPoint>(V3D(0,0,5));
    manager.visual_submap->voxel_points={unsafe.get(),safe.get()};
    manager.visual_submap->search_levels={2,0};
    manager.visual_submap->warp_patch.assign(2,std::vector<float>(256,128.f));
    manager.visual_submap->inv_expo_list={1.,1.};manager.visual_submap->errors.resize(2);
    // A padded ROI checks that valid samples use Mat::step, not camera width.
    cv::Mat padded(480,672,CV_8UC1,cv::Scalar(0));
    cv::Mat image=padded(cv::Rect(0,0,640,480));
    for (int y=0;y<image.rows;++y) for (int x=0;x<image.cols;++x)
      image.at<unsigned char>(y,x)=static_cast<unsigned char>((x/4+y/4)%256);
    float reference[64];manager.getImagePatch(image,V2D(320,240),reference,0);
    for (int x=0;x<8;++x) for (int y=0;y<8;++y)
      require(reference[x*8+y]==image.at<unsigned char>(240+x-4,320+y-4), "reference patch ignored padded image stride");
    bool rejected=false;
    try {manager.getImagePatch(image,V2D(1,1),reference,0);} catch(const std::out_of_range&) {rejected=true;}
    require(rejected,"invalid reference patch was read");
    for (int level=0;level<4;++level) for (int x=0;x<8;++x) for (int y=0;y<8;++y)
      manager.visual_submap->warp_patch[1][level*64+x*8+y]=
        image.at<unsigned char>(240+(x-4)*(1<<level),320+(y-4)*(1<<level));
    require(manager.updateState(image,3), "valid patch update failed");
    require(manager.invalid_patch_evaluations==1, "unsafe coarse patch was not omitted");
    require(state.pos_end.norm()<1e-12 && (state.rot_end-M3D::Identity()).norm()<1e-12, "invalid patch changed the state");
    manager.total_points=1;manager.visual_submap->voxel_points={unsafe.get()};
    manager.visual_submap->search_levels={2};manager.temporal_pixels=0;
    require(manager.updateState(image,3), "empty valid-patch set was not handled");
    require(manager.temporal_pixels==0, "invalid patch created an image likelihood");
    std::cout<<"visual patch footprint and frontend omission tests passed\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
