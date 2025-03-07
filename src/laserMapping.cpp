// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <chrono>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include "std_srvs/srv/trigger.hpp"
#include "fast_lio/srv/load_map.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <pcl/registration/icp.h>
#include <iomanip>
#include <sstream>
#include "scan_context.h"
#include <filesystem>  // Add this include for std::filesystem
#include <sensor_msgs/msg/laser_scan.hpp>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool    is_first_lidar = true;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

bool relocalization_mode; 
bool periodic_save;
double save_interval;
std::vector<double> initial_pose;
std::vector<PoseInfo> path_record;

rclcpp::TimerBase::SharedPtr map_save_timer_;
rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_map_service_;
rclcpp::Service<fast_lio::srv::LoadMap>::SharedPtr load_map_service_;

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubGlobalMap_;
rclcpp::TimerBase::SharedPtr global_map_timer_;

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLocalMap_;
rclcpp::TimerBase::SharedPtr local_map_timer_;
PointCloudXYZI::Ptr accumulated_cloud_;
rclcpp::Time last_local_map_time_;
std::mutex local_map_mutex_;
std::deque<std::pair<rclcpp::Time, PointCloudXYZI::Ptr>> timed_cloud_queue_; // Queue to store clouds with timestamps

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap_;

void SigHandle(int sig)
{
    flg_exit = true;
    std::cout << "catch sig %d" << sig << std::endl;
    sig_buffer.notify_all();
    rclcpp::shutdown();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if (is_first_lidar)
    {
        is_first_lidar = false;
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(cur_time);
    last_timestamp_lidar = cur_time;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::UniquePtr msg) 
{
    mtx_buffer.lock();
    double cur_time = get_time_sec(msg->header.stamp);
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (!is_first_lidar && cur_time < last_timestamp_lidar)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        lidar_buffer.clear();
    }
    if(is_first_lidar)
    {
        is_first_lidar = false;
    }
    last_timestamp_lidar = cur_time;
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in)
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        rclcpp::Time(timediff_lidar_wrt_imu + get_time_sec(msg_in->header.stamp));
    }

    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        std::cerr << "lidar loop back, clear buffer" << std::endl;
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            std::cerr << "Too few input point cloud!\n";
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = get_time_sec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
bool suppress_point_warnings = false; // Add this flag to control warnings
int warning_counter = 0; // Counter for reducing warning frequency
int info_message_counter = 0; // Counter for reducing normal info messages
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            
            // Relax the condition for filtering points
            if (points_near.size() >= 5) {
                // Check the distance to the nearest point
                float dist = calc_dist(feats_down_world->points[i], points_near[0]);
                if (dist < 0.05) {  // Reduce from 0.1 to 0.05
                    need_add = false;
                }
            }
            
            if (need_add) {
                PointToAdd.push_back(feats_down_world->points[i]);
            }
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;

    warning_counter++;
    info_message_counter++;
    
    if (feats_down_size > add_point_size/2 && !suppress_point_warnings && warning_counter >= 50)
    {
        printf("Warning: Only %d/%d points added to map! Possible filtering issue.\n", 
                add_point_size, feats_down_size);
        warning_counter = 0; // Reset counter after showing warning
    }
    else
    {
        // Only show "Added points" messages every 200 frames to reduce console spam
        if (add_point_size > 0 && info_message_counter >= 200) {
            printf("Added %d points to map from %d downsampled points\n", 
                    add_point_size, feats_down_size);
            info_message_counter = 0; // Reset counter after showing info
        }
    }
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI());
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        // laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will largely influence the real-time performences **/
    /*
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
    */
}

void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map)
{
    if (pub_map->get_subscription_count() == 0)
        return;
        
    PointCloudXYZI::Ptr scan_publish_cloud(new PointCloudXYZI());
    PointVector ikdtree_points;
    ikdtree.flatten(ikdtree.Root_Node, ikdtree_points, NOT_RECORD);
    
    // Also get deleted points back to get a complete map
    PointVector removed_points;
    ikdtree.acquire_removed_points(removed_points);
    
    // Add current scan points and removed points to the map
    scan_publish_cloud->points = ikdtree_points;
    scan_publish_cloud->points.insert(scan_publish_cloud->points.end(), removed_points.begin(), removed_points.end());
    
    sensor_msgs::msg::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*scan_publish_cloud, laserCloudMap);
    laserCloudMap.header.stamp = rclcpp::Clock().now();
    laserCloudMap.header.frame_id = "map";
    pub_map->publish(laserCloudMap);
    
    if (scan_publish_cloud->points.size() > 0) {
        printf("Publishing map with %lu points\n", scan_publish_cloud->points.size());
    }
}

void save_to_pcd()
{
    // This method is deprecated and maintained for backward compatibility
    // It's better to use the LaserMappingNode::saveMapService or LaserMappingNode::saveMapCallback methods
    RCLCPP_WARN(rclcpp::get_logger("rclcpp"), "save_to_pcd() is deprecated. Use service calls or periodic saving instead.");
    
    // Ensure PCD directory exists
    std::string pcd_dir = string(ROOT_DIR) + "PCD/";
    if (!std::filesystem::exists(pcd_dir)) {
        std::filesystem::create_directories(pcd_dir);
    }
    
    // Save to the configured path (typically test.pcd) for backward compatibility
    pcl::PCDWriter pcd_writer;
    pcd_writer.writeBinary(map_file_path, *pcl_wait_pub);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped, std::unique_ptr<tf2_ros::TransformBroadcaster> & tf_br)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "camera_init";
    trans.header.stamp = odomAftMapped.header.stamp;
    trans.child_frame_id = "body";
    trans.transform.translation.x = odomAftMapped.pose.pose.position.x;
    trans.transform.translation.y = odomAftMapped.pose.pose.position.y;
    trans.transform.translation.z = odomAftMapped.pose.pose.position.z;
    trans.transform.rotation.w = odomAftMapped.pose.pose.orientation.w;
    trans.transform.rotation.x = odomAftMapped.pose.pose.orientation.x;
    trans.transform.rotation.y = odomAftMapped.pose.pose.orientation.y;
    trans.transform.rotation.z = odomAftMapped.pose.pose.orientation.z;
    tf_br->sendTransform(trans);

    // Record path for map saving
    PoseInfo current_pose;
    current_pose.time = Measures.lidar_beg_time;
    current_pose.x = state_point.pos(0);
    current_pose.y = state_point.pos(1);
    current_pose.z = state_point.pos(2);
    current_pose.qw = state_point.rot.coeffs()[3];
    current_pose.qx = state_point.rot.coeffs()[0];
    current_pose.qy = state_point.rot.coeffs()[1];
    current_pose.qz = state_point.rot.coeffs()[2];
    path_record.push_back(current_pose);
}

void publish_path(rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time); // ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        std::cerr << "No Effective Points!" << std::endl;
        // ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

class LaserMappingNode : public rclcpp::Node
{
public:
    LaserMappingNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions()) : Node("laser_mapping", options)
    {
        this->declare_parameter<bool>("publish.path_en", true);
        this->declare_parameter<bool>("publish.effect_map_en", false);
        this->declare_parameter<bool>("publish.map_en", false);
        this->declare_parameter<bool>("publish.scan_publish_en", true);
        this->declare_parameter<bool>("publish.dense_publish_en", true);
        this->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
        this->declare_parameter<int>("max_iteration", 4);
        this->declare_parameter<string>("map_file_path", "");
        this->declare_parameter<string>("common.lid_topic", "/livox/lidar");
        this->declare_parameter<string>("common.imu_topic", "/livox/imu");
        this->declare_parameter<bool>("common.time_sync_en", false);
        this->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
        this->declare_parameter<double>("filter_size_corner", 0.5);
        this->declare_parameter<double>("filter_size_surf", 0.5);
        this->declare_parameter<double>("filter_size_map", 0.5);
        this->declare_parameter<double>("cube_side_length", 200.);
        this->declare_parameter<float>("mapping.det_range", 300.);
        this->declare_parameter<double>("mapping.fov_degree", 180.);
        this->declare_parameter<double>("mapping.gyr_cov", 0.1);
        this->declare_parameter<double>("mapping.acc_cov", 0.1);
        this->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
        this->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
        this->declare_parameter<double>("preprocess.blind", 0.01);
        this->declare_parameter<int>("preprocess.lidar_type", AVIA);
        this->declare_parameter<int>("preprocess.scan_line", 16);
        this->declare_parameter<int>("preprocess.timestamp_unit", US);
        this->declare_parameter<int>("preprocess.scan_rate", 10);
        this->declare_parameter<int>("point_filter_num", 2);
        this->declare_parameter<bool>("feature_extract_enable", false);
        this->declare_parameter<bool>("runtime_pos_log_enable", false);
        this->declare_parameter<bool>("mapping.extrinsic_est_en", true);
        this->declare_parameter<bool>("pcd_save.pcd_save_en", false);
        this->declare_parameter<int>("pcd_save.interval", -1);
        this->declare_parameter<vector<double>>("mapping.extrinsic_T", vector<double>());
        this->declare_parameter<vector<double>>("mapping.extrinsic_R", vector<double>());
        this->declare_parameter("mapping.relocalization_mode", false);
        this->declare_parameter("mapping.initial_pose", std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
        this->declare_parameter("pcd_save.periodic_save", false);
        this->declare_parameter("pcd_save.save_interval", 60.0);
        this->declare_parameter("mapping.suppress_warnings", false);
        this->declare_parameter("mapping.max_height", 3.0);
        this->declare_parameter("mapping.ground_level", -0.5);
        
        this->get_parameter_or<bool>("publish.path_en", path_en, true);
        this->get_parameter_or<bool>("publish.effect_map_en", effect_pub_en, false);
        this->get_parameter_or<bool>("publish.map_en", map_pub_en, false);
        this->get_parameter_or<bool>("publish.scan_publish_en", scan_pub_en, true);
        this->get_parameter_or<bool>("publish.dense_publish_en", dense_pub_en, true);
        this->get_parameter_or<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
        this->get_parameter_or<int>("max_iteration", NUM_MAX_ITERATIONS, 4);
        this->get_parameter_or<string>("map_file_path", map_file_path, "");
        this->get_parameter_or<string>("common.lid_topic", lid_topic, "/livox/lidar");
        this->get_parameter_or<string>("common.imu_topic", imu_topic,"/livox/imu");
        this->get_parameter_or<bool>("common.time_sync_en", time_sync_en, false);
        this->get_parameter_or<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
        this->get_parameter_or<double>("filter_size_corner",filter_size_corner_min,0.5);
        this->get_parameter_or<double>("filter_size_surf",filter_size_surf_min,0.5);
        this->get_parameter_or<double>("filter_size_map",filter_size_map_min,0.5);
        this->get_parameter_or<double>("cube_side_length",cube_len,200.f);
        this->get_parameter_or<float>("mapping.det_range",DET_RANGE,300.f);
        this->get_parameter_or<double>("mapping.fov_degree",fov_deg,180.f);
        this->get_parameter_or<double>("mapping.gyr_cov",gyr_cov,0.1);
        this->get_parameter_or<double>("mapping.acc_cov",acc_cov,0.1);
        this->get_parameter_or<double>("mapping.b_gyr_cov",b_gyr_cov,0.0001);
        this->get_parameter_or<double>("mapping.b_acc_cov",b_acc_cov,0.0001);
        this->get_parameter_or<double>("preprocess.blind", p_pre->blind, 0.01);
        this->get_parameter_or<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
        this->get_parameter_or<int>("preprocess.scan_line", p_pre->N_SCANS, 16);
        this->get_parameter_or<int>("preprocess.timestamp_unit", p_pre->time_unit, US);
        this->get_parameter_or<int>("preprocess.scan_rate", p_pre->SCAN_RATE, 10);
        this->get_parameter_or<int>("point_filter_num", p_pre->point_filter_num, 2);
        this->get_parameter_or<bool>("feature_extract_enable", p_pre->feature_enabled, false);
        this->get_parameter_or<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
        this->get_parameter_or<bool>("mapping.extrinsic_est_en", extrinsic_est_en, true);
        this->get_parameter_or<bool>("pcd_save.pcd_save_en", pcd_save_en, false);
        this->get_parameter_or<int>("pcd_save.interval", pcd_save_interval, -1);
        this->get_parameter_or<vector<double>>("mapping.extrinsic_T", extrinT, vector<double>());
        this->get_parameter_or<vector<double>>("mapping.extrinsic_R", extrinR, vector<double>());
        this->get_parameter_or<bool>("mapping.relocalization_mode", relocalization_mode, false);
        this->get_parameter_or<vector<double>>("mapping.initial_pose", initial_pose, vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0});
        this->get_parameter_or<bool>("pcd_save.periodic_save", periodic_save, false);
        this->get_parameter_or<double>("pcd_save.save_interval", save_interval, 60.0);
        this->get_parameter_or<bool>("mapping.suppress_warnings", suppress_point_warnings, false);
        this->get_parameter_or<double>("mapping.max_height", max_height_, 3.0);
        this->get_parameter_or<double>("mapping.ground_level", ground_level_, -0.5);
        
        // For sequential scan TOF lidars, these warnings are expected, so suppress them by default
        if (p_pre->lidar_type == AVIA) {
            RCLCPP_INFO(this->get_logger(), "Sequential scan TOF lidar detected, suppressing point filter warnings by default");
            suppress_point_warnings = true;
        }

        RCLCPP_INFO(this->get_logger(), "p_pre->lidar_type %d", p_pre->lidar_type);

        path.header.stamp = this->get_clock()->now();
        path.header.frame_id ="camera_init";

        FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
        HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

        _featsArray.reset(new PointCloudXYZI());

        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
        memset(point_selected_surf, true, sizeof(point_selected_surf));
        memset(res_last, -1000.0f, sizeof(res_last));

        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
        p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
        p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
        p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
        p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

        fill(epsi, epsi+23, 0.001);
        kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

        string pos_log_dir = root_dir + "/Log/pos_log.txt";
        fp = fopen(pos_log_dir.c_str(),"w");

        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
        fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
        if (fout_pre && fout_out)
            cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
        else
            cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

        if (p_pre->lidar_type == AVIA)
        {
            sub_pcl_livox_ = this->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
        }
        else
        {
            sub_pcl_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
        }
        sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(imu_topic, 10, imu_cbk);
        pubLaserCloudFull_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 20);
        pubLaserCloudFull_body_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 20);
        pubLaserCloudEffect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 20);
        pubOdomAftMapped_ = this->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 20);
        pubPath_ = this->create_publisher<nav_msgs::msg::Path>("/path", 20);
        pubLaserScan_ = this->create_publisher<sensor_msgs::msg::LaserScan>("/scan", 20); // Changed to standard /scan topic
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        auto period_ms = std::chrono::milliseconds(static_cast<int64_t>(1000.0 / 100.0));
        timer_ = rclcpp::create_timer(this, this->get_clock(), period_ms, std::bind(&LaserMappingNode::timer_callback, this));

        map_save_srv_ = this->create_service<std_srvs::srv::Trigger>("map_save", std::bind(&LaserMappingNode::map_save_callback, this, std::placeholders::_1, std::placeholders::_2));

        if (periodic_save && save_interval > 0) {
            map_save_timer_ = this->create_wall_timer(
                std::chrono::duration<double>(save_interval),
                std::bind(&LaserMappingNode::saveMapCallback, this));
            RCLCPP_INFO(this->get_logger(), "Periodic map saving enabled, interval: %.1f seconds", save_interval);
        }

        save_map_service_ = this->create_service<std_srvs::srv::Trigger>(
            "save_map", std::bind(&LaserMappingNode::saveMapService, this, std::placeholders::_1, std::placeholders::_2));
        
        load_map_service_ = this->create_service<fast_lio::srv::LoadMap>(
            "load_map", std::bind(&LaserMappingNode::loadMapService, this, std::placeholders::_1, std::placeholders::_2));

        pubGlobalMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/global_map", 1);
        global_map_timer_ = this->create_wall_timer(
            std::chrono::seconds(5), 
            std::bind(&LaserMappingNode::publishGlobalMap, this));

        // Initialize local map accumulation
        accumulated_cloud_.reset(new PointCloudXYZI());
        last_local_map_time_ = this->get_clock()->now();
        timed_cloud_queue_.clear(); // Initialize the timed cloud queue
        
        // Create publisher for local map
        pubLocalMap_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/local_map", 1);
        
        // Create timer for local map publishing (10Hz)
        local_map_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100), 
            std::bind(&LaserMappingNode::publishLocalMap, this));
        
        // After node initialization and parameter loading, before RCLCPP_INFO about node init finished
        // Add automatic map loading and relocalization
        std::string pcd_dir = string(ROOT_DIR) + "PCD/";
        std::string default_map_path = pcd_dir + "global_map.pcd";
        if (std::filesystem::exists(default_map_path)) {
            RCLCPP_INFO(this->get_logger(), "Found existing map at %s, attempting to load it", default_map_path.c_str());
            bool load_success = loadMap(default_map_path);
            if (load_success) {
                RCLCPP_INFO(this->get_logger(), "Map loaded successfully, enabling relocalization mode");
                // Enable relocalization automatically when map is loaded
                relocalization_mode = true;
                
                // In relocalization mode, we don't set the initial pose here
                // The system will automatically determine the position using accumulated scans
                // when enough measurements have been received to perform place recognition
                RCLCPP_INFO(this->get_logger(), "Automatic relocalization will be performed when enough measurements are received");
            } else {
                RCLCPP_WARN(this->get_logger(), "Failed to load map, continuing without relocalization");
            }
        }
        
        RCLCPP_INFO(this->get_logger(), "Node init finished.");

        if (relocalization_mode) {
            RCLCPP_INFO(this->get_logger(), "Starting in re-localization mode");
            // The initial pose will be determined through scan context matching
            // The first few scans will be used to generate a scan context descriptor
            // which will be matched against the map to find the initial position
        }

        // Adjust downsampling parameters
        filter_size_map_min = this->declare_parameter("mapping.filter_size_map", 0.1); // Reduce from 0.2 to 0.1
        filter_size_surf_min = this->declare_parameter("mapping.filter_size_surf", 0.1); // Reduce if needed
        
        // Update voxel grid filter settings
        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
        downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

        // Loop closure parameters
        this->declare_parameter<bool>("loop_closure.enabled", true);
        this->declare_parameter<double>("loop_closure.search_radius", LOOP_CLOSURE_SEARCH_RADIUS);
        this->declare_parameter<double>("loop_closure.min_distance", LOOP_CLOSURE_MIN_DIST);
        this->declare_parameter<int>("loop_closure.detection_interval", LOOP_CLOSURE_DETECTION_INTERVAL);
        this->declare_parameter<double>("loop_closure.fitness_score_threshold", LOOP_CLOSURE_FITNESS_SCORE_THRESH);
        
        loop_closure_enabled_ = this->get_parameter("loop_closure.enabled").as_bool();
        loop_closure_search_radius_ = this->get_parameter("loop_closure.search_radius").as_double();
        loop_closure_min_distance_ = this->get_parameter("loop_closure.min_distance").as_double();
        loop_closure_detection_interval_ = this->get_parameter("loop_closure.detection_interval").as_int();
        loop_closure_fitness_score_threshold_ = this->get_parameter("loop_closure.fitness_score_threshold").as_double();
        
        // Publisher for loop closure markers
        pubLoopClosureMarker_ = this->create_publisher<visualization_msgs::msg::Marker>("loop_closure_marker", 10);
        
        RCLCPP_INFO(this->get_logger(), "Loop closure detection %s", loop_closure_enabled_ ? "enabled" : "disabled");
        if (loop_closure_enabled_) {
            RCLCPP_INFO(this->get_logger(), "Loop closure parameters:");
            RCLCPP_INFO(this->get_logger(), "  Search radius: %.2f m", loop_closure_search_radius_);
            RCLCPP_INFO(this->get_logger(), "  Min distance: %.2f m", loop_closure_min_distance_);
            RCLCPP_INFO(this->get_logger(), "  Detection interval: %d frames", loop_closure_detection_interval_);
            RCLCPP_INFO(this->get_logger(), "  Fitness score threshold: %.2f", loop_closure_fitness_score_threshold_);
        }

        RCLCPP_INFO(this->get_logger(), "Map height filtering: min=%f, max=%f", ground_level_, max_height_);
    }

    ~LaserMappingNode()
    {
        fout_out.close();
        fout_pre.close();
        fclose(fp);
    }

    bool loadMap(const std::string& map_path)
    {
        pcl::PointCloud<PointType>::Ptr loaded_map(new pcl::PointCloud<PointType>());
        
        if (pcl::io::loadPCDFile<PointType>(map_path, *loaded_map) == -1) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load map from %s", map_path.c_str());
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Loaded map with %lu points from %s", 
                    loaded_map->size(), map_path.c_str());
        
        // Clear current map using DeleteTree() method
        ikdtree.InitializeKDTree(); // Reset the tree with default parameters
        
        // Insert loaded points into ikd-tree
        ikdtree.Build(loaded_map->points);
        
        // For scan context-based relocalization, create a keyframe from the loaded map
        // We'll use this as a reference for place recognition
        if (!keyframes_.empty()) {
            keyframes_.clear(); // Clear any existing keyframes
        }
        
        // Downsample the map for efficient scan context generation
        pcl::PointCloud<PointType>::Ptr downsampled_map(new pcl::PointCloud<PointType>());
        pcl::VoxelGrid<PointType> voxel_grid;
        voxel_grid.setLeafSize(0.5, 0.5, 0.5); // Larger leaf size for scan context
        voxel_grid.setInputCloud(loaded_map);
        voxel_grid.filter(*downsampled_map);
        
        // Create a keyframe for the map at origin
        // The actual pose doesn't matter as much as the scan context descriptor
        Eigen::Matrix4d map_pose = Eigen::Matrix4d::Identity();
        
        std::shared_ptr<KeyFrame> map_keyframe = std::make_shared<KeyFrame>(
            0, // ID
            0.0, // timestamp (not important for map)
            map_pose,
            downsampled_map
        );
        
        // Generate scan context for this keyframe
        map_keyframe->scan_context = generateScanContext(downsampled_map);
        
        // Add to keyframes list
        keyframes_.push_back(map_keyframe);
        
        RCLCPP_INFO(this->get_logger(), "Created keyframe from map with %lu points for scan context matching",
                   downsampled_map->size());
        
        return true;
    }

    bool loadMapService(const std::shared_ptr<fast_lio::srv::LoadMap::Request> request,
                       std::shared_ptr<fast_lio::srv::LoadMap::Response> response)
    {
        bool success = loadMap(request->map_path);
        response->success = success;
        if (success) {
            response->message = "Map loaded successfully";
        } else {
            response->message = "Failed to load map";
        }
        return true;
    }

    void timer_callback()
    {
        // Remove static declarations since we now use class members
        if (relocalization_mode && !relocalization_done_)
        {
            if (!sync_packages(Measures)) {
                return;
            }

            p_imu->Process(Measures, kf, feats_undistort);
            feats_down_body->clear();
            feats_down_world->clear();
            
            // Filter points and transform them to world frame
            for (int i = 0; i < feats_undistort->points.size(); i++)
            {
                if (i % p_pre->point_filter_num == 0)
                {
                    PointType p_body(feats_undistort->points[i]);
                    feats_down_body->points.push_back(p_body);
                    
                    PointType p_world;
                    pointBodyToWorld(&p_body, &p_world);
                    feats_down_world->points.push_back(p_world);
                }
            }
            
            // Update size for further processing
            feats_down_size = feats_down_world->points.size();
            
            // Add these points to our accumulated local map
            accumulateLocalMap(feats_down_world);
            
            // Try relocalization if we have accumulated enough points
            if (accumulated_cloud_->points.size() > 3000) {
                if (attemptRelocalization()) {
                    relocalization_done_ = true;
                    relocalization_reset_needed_ = true;
                    
                    // Clear the accumulated cloud now that relocalization succeeded
                    accumulated_cloud_->clear();
                    timed_cloud_queue_.clear(); // Also clear the timed cloud queue
                    RCLCPP_INFO(this->get_logger(), "Relocalization completed successfully. System state and map have been reset to the new position.");
                    
                    // Force a reset of the lidar_buffer and imu_buffer to avoid using outdated measurements
                    mtx_buffer.lock();
                    lidar_buffer.clear();
                    imu_buffer.clear();
                    mtx_buffer.unlock();
                    
                    // Return early to ensure the next scan starts fresh with the new position
                    return;
                } else {
                    RCLCPP_INFO(this->get_logger(), "Attempting relocalization (accumulated %zu points, need more data...)", 
                               accumulated_cloud_->points.size());
                }
            } else {
                RCLCPP_INFO(this->get_logger(), "Accumulating points for relocalization: %zu/%d", 
                           accumulated_cloud_->points.size(), 3000);
            }
            
            // Skip the rest of processing until relocalized
            if (!relocalization_done_) return;
        }

        // Handle first scan after relocalization
        if (relocalization_done_ && relocalization_reset_needed_) {
            if (!sync_packages(Measures)) {
                return;
            }
            
            // Reset this flag to indicate we've handled the relocalization reset
            relocalization_reset_needed_ = false;
            
            // Need to reinitialize first_lidar_time to avoid discontinuities
            first_lidar_time = Measures.lidar_beg_time;
            p_imu->first_lidar_time = first_lidar_time;
            
            // Set flag to force re-initialization properly using public properties 
            // instead of accessing private members
            p_imu->Reset(); // Use the public Reset() method to fully reset IMU state
            
            // Start scan processing from beginning
            flg_first_scan = true;
            flg_EKF_inited = false;
            
            RCLCPP_INFO(this->get_logger(), "Processing first scan after successful relocalization with full IMU reset");
            return; // Skip the rest of the processing to force a complete restart cycle
        }

        /*** Segment the map in lidar FOV ***/
        lasermap_fov_segment();

        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                return;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            
            p_imu->Process(Measures, kf, feats_undistort);
            
            if (feats_undistort->empty() || (feats_undistort == NULL)) {
                return;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();
            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    feats_down_world->resize(feats_down_size);
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    ikdtree.Build(feats_down_world->points);
                }
                return;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                RCLCPP_WARN(this->get_logger(), "No point, skip this scan!\n");
                return;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            
            // Add the current scan to the local map for continuous visualization
            accumulateLocalMap(feats_down_world);
            
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath_);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull_);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body_);
            if (effect_pub_en) publish_effect_world(pubLaserCloudEffect_);
            // if (map_pub_en) publish_map(pubLaserCloudMap_);
            
            // Publish laser scan for costmap usage
            if (scan_pub_en) publish_laser_scan(feats_down_body);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }

            // After processing the current frame and updating the map
            if (loop_closure_enabled_ && flg_EKF_inited) {
                // Add keyframe at regular intervals
                static int frame_count = 0;
                frame_count++;
                
                if (frame_count % loop_closure_detection_interval_ == 0) {
                    addKeyFrame();
                    detectLoopClosure();
                }
            }
        }
    }

    void map_save_callback(const std::shared_ptr<std_srvs::srv::Trigger::Request> req, std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        // Delegate to the service method that handles on-demand saving to global_map.pcd
        saveMapService(req, res);
    }

    void saveMapCallback()
    {
        if (!pcd_save_en) return;
        
        RCLCPP_INFO(this->get_logger(), "Saving periodic map snapshot...");
        
        // Ensure PCD directory exists
        std::string pcd_dir = string(ROOT_DIR) + "PCD/";
        if (!std::filesystem::exists(pcd_dir)) {
            RCLCPP_INFO(this->get_logger(), "Creating PCD directory: %s", pcd_dir.c_str());
            std::filesystem::create_directories(pcd_dir);
        }
        
        // Generate timestamped filename for periodic saves
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        std::string timestamp_str = ss.str();
        
        // Get all points from ikdtree
        PointVector ikdtree_points;
        ikdtree.flatten(ikdtree.Root_Node, ikdtree_points, NOT_RECORD);
        
        // Convert to PointCloudXYZI
        PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
        global_map->points.resize(ikdtree_points.size());
        for (size_t i = 0; i < ikdtree_points.size(); i++) {
            global_map->points[i] = ikdtree_points[i];
        }
        
        // Filter points by height
        PointCloudXYZI::Ptr filtered_map = filterPointsByHeight(global_map);
        
        // Use timestamped filename format for periodic saves
        std::string periodic_map_path(pcd_dir + "tmp/Global_Map_" + timestamp_str + ".pcd");
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(periodic_map_path, *filtered_map);
        
        // Save trajectory
        std::string trajectory_path = pcd_dir + "tmp/trajectory_" + timestamp_str + ".txt";
        std::ofstream trajectory_file(trajectory_path);
        
        if (trajectory_file.is_open()) {
            for (const auto& pose : path_record) {
                trajectory_file << std::fixed << std::setprecision(6)
                              << pose.time << " "
                              << pose.x << " " << pose.y << " " << pose.z << " "
                              << pose.qw << " " << pose.qx << " " << pose.qy << " " << pose.qz << std::endl;
            }
            trajectory_file.close();
        }
        
        RCLCPP_INFO(this->get_logger(), "Saved periodic map with %zu points (filtered from %zu points) to %s",
                    filtered_map->points.size(), global_map->points.size(), periodic_map_path.c_str());
    }

    bool saveMapService(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                       std::shared_ptr<std_srvs::srv::Trigger::Response> response)
    {
        RCLCPP_INFO(this->get_logger(), "Saving map to PCD file...");
        
        // Ensure PCD directory exists
        std::string pcd_dir = string(ROOT_DIR) + "PCD/";
        if (!std::filesystem::exists(pcd_dir)) {
            RCLCPP_INFO(this->get_logger(), "Creating PCD directory: %s", pcd_dir.c_str());
            std::filesystem::create_directories(pcd_dir);
        }
        
        // Generate timestamped filename
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
        std::string timestamp_str = ss.str();
        
        // Get all points from ikdtree
        PointVector ikdtree_points;
        ikdtree.flatten(ikdtree.Root_Node, ikdtree_points, NOT_RECORD);
        
        // Convert to PointCloudXYZI
        PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
        global_map->points.resize(ikdtree_points.size());
        for (size_t i = 0; i < ikdtree_points.size(); i++) {
            global_map->points[i] = ikdtree_points[i];
        }
        
        // Filter points by height
        PointCloudXYZI::Ptr filtered_map = filterPointsByHeight(global_map);
        
        // Save trajectory
        std::string trajectory_path = pcd_dir + "tmp/trajectory_" + timestamp_str + ".txt";
        std::ofstream trajectory_file(trajectory_path);
        
        if (trajectory_file.is_open()) {
            for (const auto& pose : path_record) {
                trajectory_file << std::fixed << std::setprecision(6)
                               << pose.time << " "
                               << pose.x << " " << pose.y << " " << pose.z << " "
                               << pose.qw << " " << pose.qx << " " << pose.qy << " " << pose.qz << std::endl;
            }
            trajectory_file.close();
            RCLCPP_INFO(this->get_logger(), "Saved trajectory to %s", trajectory_path.c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "Failed to open trajectory file for writing: %s", trajectory_path.c_str());
        }
        
        // Save the map
        std::string pcd_path = pcd_dir + "tmp/Global_Map_" + timestamp_str + ".pcd";
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(pcd_path, *filtered_map);
        
        response->success = true;
        response->message = "Successfully saved map with " + std::to_string(filtered_map->points.size()) + 
                          " points (filtered from " + std::to_string(global_map->points.size()) + " points) to " + pcd_path;
        
        RCLCPP_INFO(this->get_logger(), "%s", response->message.c_str());
        return true;
    }

    void publishGlobalMap() {
        if (pubGlobalMap_->get_subscription_count() == 0) {
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Publishing global map...");

        // Get all points from ikdtree
        PointVector ikdtree_points;
        ikdtree.flatten(ikdtree.Root_Node, ikdtree_points, NOT_RECORD);
        
        // Convert to PointCloudXYZI
        PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
        global_map->points.resize(ikdtree_points.size());
        for (size_t i = 0; i < ikdtree_points.size(); i++) {
            global_map->points[i] = ikdtree_points[i];
        }
        
        // Filter points by height for the output map
        PointCloudXYZI::Ptr filtered_map = filterPointsByHeight(global_map);
        
        // Convert to ROS message and publish
        sensor_msgs::msg::PointCloud2 global_map_msg;
        pcl::toROSMsg(*filtered_map, global_map_msg);
        global_map_msg.header.stamp = this->get_clock()->now();
        global_map_msg.header.frame_id = "map";
        pubGlobalMap_->publish(global_map_msg);
        
        RCLCPP_INFO(this->get_logger(), "Global map published with %zu points (filtered from %zu points)",
                   filtered_map->points.size(), global_map->points.size());
    }

    // Add current frame as a keyframe
    void addKeyFrame() {
        // Create a copy of the current point cloud
        pcl::PointCloud<PointType>::Ptr cloud_copy(new pcl::PointCloud<PointType>());
        *cloud_copy = *feats_down_body;
        
        // Create transformation matrix from current state
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        Eigen::Matrix3d rot = state_point.rot.toRotationMatrix();
        
        // Fill rotation part
        pose.block<3, 3>(0, 0) = rot;
        
        // Fill translation part
        pose.block<3, 1>(0, 3) = Eigen::Vector3d(state_point.pos(0), state_point.pos(1), state_point.pos(2));
        
        // Create keyframe
        std::shared_ptr<KeyFrame> keyframe = std::make_shared<KeyFrame>(
            keyframes_.size(),
            Measures.lidar_beg_time,
            pose,
            cloud_copy
        );
        
        // Generate scan context for this keyframe
        keyframe->scan_context = generateScanContext(cloud_copy);
        
        // Add to keyframes list
        keyframes_.push_back(keyframe);
        
        RCLCPP_INFO(this->get_logger(), "Added keyframe %d at position [%.2f, %.2f, %.2f]", 
                    static_cast<int>(keyframes_.size() - 1),
                    state_point.pos(0), state_point.pos(1), state_point.pos(2));
    }
    
    // Detect loop closure
    void detectLoopClosure() {
        if (keyframes_.size() < 2) {
            return; // Not enough keyframes yet
        }
        
        // Get the latest keyframe
        std::shared_ptr<KeyFrame> current_keyframe = keyframes_.back();
        
        // Get current position
        Eigen::Vector3d current_position = current_keyframe->pose.block<3, 1>(0, 3);
        
        // Exclude the most recent keyframes (exclude last loop_closure_detection_interval)
        int exclude_recent_num = 1;
        
        // Find the best matching scan context
        auto [match_idx, match_score] = findBestMatchingScanContext(
            current_keyframe->scan_context,
            keyframes_,
            exclude_recent_num
        );
        
        // If no match found
        if (match_idx < 0) {
            return;
        }
        
        // Get the matched keyframe
        std::shared_ptr<KeyFrame> matched_keyframe = keyframes_[match_idx];
        
        // Get matched position
        Eigen::Vector3d matched_position = matched_keyframe->pose.block<3, 1>(0, 3);
        
        // Check if the keyframes are far enough apart
        double distance = (current_position - matched_position).norm();
        if (distance < loop_closure_min_distance_) {
            return; // Too close, not a true loop closure
        }
        
        // Perform ICP to refine the transformation
        double fitness_score = 0.0;
        Eigen::Matrix4d relative_transform = performICPMatching(
            current_keyframe->cloud,
            matched_keyframe->cloud,
            Eigen::Matrix4d::Identity(), // Initial guess (can be improved)
            &fitness_score
        );
        
        // Check ICP fitness score
        if (fitness_score > loop_closure_fitness_score_threshold_) {
            RCLCPP_INFO(this->get_logger(), "Loop closure ICP failed: fitness score too high (%.4f > %.4f)",
                       fitness_score, loop_closure_fitness_score_threshold_);
            return;
        }
        
        // Compute the error/drift
        Eigen::Matrix4d expected_pose = matched_keyframe->pose;
        Eigen::Matrix4d actual_pose = current_keyframe->pose;
        Eigen::Matrix4d error = expected_pose.inverse() * actual_pose * relative_transform.inverse();
        
        // Apply correction (simplified - in a full implementation, we would update the pose graph)
        RCLCPP_INFO(this->get_logger(), "Loop closure detected between frames %d and %d (current), distance: %.2f m, score: %.4f",
                    match_idx, static_cast<int>(keyframes_.size() - 1), distance, match_score);
        
        // Publish marker for visualization
        publishLoopClosureMarker(matched_position, current_position);
        
        // In a real implementation, you would:
        // 1. Update the pose graph
        // 2. Perform pose graph optimization
        // 3. Update all poses
        // 4. Rebuild the map
        
        // For demonstration, we'll just print the correction
        Eigen::Vector3d translation_error = error.block<3, 1>(0, 3);
        RCLCPP_INFO(this->get_logger(), "Detected drift: [%.2f, %.2f, %.2f] m",
                    translation_error.x(), translation_error.y(), translation_error.z());
    }
    
    // Publish a marker to visualize the loop closure
    void publishLoopClosureMarker(const Eigen::Vector3d& position1, const Eigen::Vector3d& position2) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "loop_closure";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2; // Line width
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;
        
        geometry_msgs::msg::Point p1, p2;
        p1.x = position1.x();
        p1.y = position1.y();
        p1.z = position1.z();
        
        p2.x = position2.x();
        p2.y = position2.y();
        p2.z = position2.z();
        
        marker.points.push_back(p1);
        marker.points.push_back(p2);
        
        pubLoopClosureMarker_->publish(marker);
    }

    bool attemptRelocalization() 
    {
        // Check if we have loaded a map keyframe
        if (keyframes_.size() < 1) {
            RCLCPP_INFO(this->get_logger(), "Cannot attempt relocalization, no keyframes loaded (%zu keyframes, %zu points)", 
                       keyframes_.size(), accumulated_cloud_->points.size());
            return false;
        }
        
        // Ensure we have enough points in the accumulated cloud for a robust match
        // Increased from 1000 to 3000 points for better relocalization in new positions
        if (accumulated_cloud_->points.size() < 3000) {
            RCLCPP_INFO(this->get_logger(), "Not enough points for relocalization (%zu points, need at least 3000)", 
                       accumulated_cloud_->points.size());
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Attempting relocalization with accumulated cloud of %zu points", 
                   accumulated_cloud_->points.size());
        
        // Get the first keyframe (the map)
        std::shared_ptr<KeyFrame> map_keyframe = keyframes_[0];
        
        // Use the accumulated cloud
        PointCloudXYZI::Ptr source_cloud(new PointCloudXYZI(*accumulated_cloud_));
        PointCloudXYZI::Ptr target_cloud(new PointCloudXYZI(*map_keyframe->cloud));
        
        // Filter clouds for faster ICP - use a safe leaf size to avoid integer overflow
        PointCloudXYZI::Ptr source_filtered(new PointCloudXYZI());
        PointCloudXYZI::Ptr target_filtered(new PointCloudXYZI());
        
        // Temporarily set a larger leaf size to avoid the integer overflow issue
        double original_leaf_size = filter_size_map_min;
        double safe_leaf_size = std::max(0.2, original_leaf_size); // Use at least 0.2m leaf size
        
        pcl::VoxelGrid<PointType> temp_filter;
        temp_filter.setLeafSize(safe_leaf_size, safe_leaf_size, safe_leaf_size);
        
        // Filter source and target using the safer parameters
        temp_filter.setInputCloud(source_cloud);
        temp_filter.filter(*source_filtered);
        
        temp_filter.setInputCloud(target_cloud);
        temp_filter.filter(*target_filtered);
        
        RCLCPP_INFO(this->get_logger(), "Filtered source cloud: %zu points, target cloud: %zu points",
                  source_filtered->points.size(), target_filtered->points.size());
        
        // Define a list of initial transformations to try
        // This helps ICP find the correct alignment even with large displacements
        std::vector<Eigen::Matrix4f> initial_transformations;
        
        // Add identity matrix (no transformation - default)
        Eigen::Matrix4f identity = Eigen::Matrix4f::Identity();
        initial_transformations.push_back(identity);
        
        // Add translation of 5m along X (forward)
        Eigen::Matrix4f trans_5m_x = Eigen::Matrix4f::Identity();
        trans_5m_x(0, 3) = 5.0;  // 5m in x direction
        initial_transformations.push_back(trans_5m_x);
        
        // Add translation of 5m along Y (right)
        Eigen::Matrix4f trans_5m_y = Eigen::Matrix4f::Identity();
        trans_5m_y(1, 3) = 5.0;  // 5m in y direction
        initial_transformations.push_back(trans_5m_y);
        
        // Add translation of -5m along Y (left)
        Eigen::Matrix4f trans_neg_5m_y = Eigen::Matrix4f::Identity();
        trans_neg_5m_y(1, 3) = -5.0;  // -5m in y direction
        initial_transformations.push_back(trans_neg_5m_y);
        
        // Also try 10m in x direction (based on the screenshot showing ~10m displacement)
        Eigen::Matrix4f trans_10m_x = Eigen::Matrix4f::Identity();
        trans_10m_x(0, 3) = 10.0;
        initial_transformations.push_back(trans_10m_x);
        
        // Store best result
        bool converged = false;
        float best_fitness_score = std::numeric_limits<float>::max();
        Eigen::Matrix4f best_transformation = Eigen::Matrix4f::Identity();
        
        // Try each initial transformation
        for (size_t i = 0; i < initial_transformations.size(); i++) {
            // Perform ICP to find the transformation
            pcl::IterativeClosestPoint<PointType, PointType> icp;
            // Increase max correspondence distance to handle larger displacements
            icp.setMaxCorrespondenceDistance(10.0);
            // Increase max iterations for more thorough alignment
            icp.setMaximumIterations(200);
            icp.setTransformationEpsilon(1e-6);
            icp.setEuclideanFitnessEpsilon(1e-6);
            
            // Set initial transformation
            icp.setInputSource(source_filtered);
            icp.setInputTarget(target_filtered);
            
            // Set initial alignment
            Eigen::Matrix4f init_guess = initial_transformations[i];
            
            RCLCPP_INFO(this->get_logger(), "Trying initial transformation #%zu", i);
            
            PointCloudXYZI::Ptr aligned(new PointCloudXYZI());
            icp.align(*aligned, init_guess);
            
            float fitness_score = icp.getFitnessScore();
            RCLCPP_INFO(this->get_logger(), "Initial guess %zu: Converged=%d, Fitness score=%f", 
                       i, icp.hasConverged(), fitness_score);
            
            // Update best result if this one is better
            if (icp.hasConverged() && fitness_score < best_fitness_score) {
                converged = true;
                best_fitness_score = fitness_score;
                best_transformation = icp.getFinalTransformation();
            }
        }
        
        // Check if any of the attempts converged with a good fitness score
        if (converged && best_fitness_score < 5.0) {
            // Extract transformation parameters
            Eigen::Matrix3f rotation_matrix = best_transformation.block<3,3>(0,0);
            Eigen::Vector3f translation = best_transformation.block<3,1>(0,3);
            
            // Convert to quaternion
            Eigen::Quaternionf q(rotation_matrix);
            
            RCLCPP_INFO(this->get_logger(), "Relocalization successful! Position: [%f, %f, %f], Orientation (quat): [%f, %f, %f, %f], Fitness score: %f", 
                       translation(0), translation(1), translation(2),
                       q.w(), q.x(), q.y(), q.z(),
                       best_fitness_score);
            
            // Update the state with the found position
            state_point.pos = V3D(translation(0), translation(1), translation(2));
            state_point.rot = Eigen::Quaterniond(q.w(), q.x(), q.y(), q.z());
            
            // Reset the velocity and bias states - these should be re-estimated from the new position
            state_point.vel.setZero();
            state_point.bg.setZero();
            state_point.ba.setZero();
            
            // Update derived state variables used for odometry and mapping
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];
            
            // Reset KF filter state to match the new position
            kf.change_x(state_point);
            
            // Reset EKF covariance to initial values
            esekfom::esekf<state_ikfom, 12, input_ikfom>::cov P = kf.get_P();
            P.setIdentity();
            P(6,6) = P(7,7) = P(8,8) = 0.00001;
            P(9,9) = P(10,10) = P(11,11) = 0.00001;
            P(15,15) = P(16,16) = P(17,17) = 0.0001;
            P(18,18) = P(19,19) = P(20,20) = 0.001;
            P(21,21) = P(22,22) = 0.00001;
            kf.change_P(P);
            
            // Clear the path to start from the relocated position
            path.poses.clear();
            
            // Clear the map and create a new tree with the keyframe data
            ikdtree.InitializeKDTree();
            
            // Use the first keyframe cloud as the initial map
            PointCloudXYZI::Ptr initial_map(new PointCloudXYZI(*map_keyframe->cloud));
            ikdtree.Build(initial_map->points);
            
            // Completely reset the IMU processor to avoid velocity discontinuities
            p_imu->Reset();
            
            // Remove these lines that reference the local variables that no longer exist
            // relocalization_done = true;
            // relocalization_reset_needed = true;
            
            // Reset internal variables to ensure clean restart
            flg_EKF_inited = false;
            
            // Immediately publish the updated odometry to reflect the new position
            publish_odometry(pubOdomAftMapped_, tf_broadcaster_);
            
            return true;
        } else {
            RCLCPP_WARN(this->get_logger(), "All ICP attempts failed to find a good match. Best fitness score: %f", 
                       best_fitness_score);
            return false;
        }
    }

    void publishLocalMap() {
        std::lock_guard<std::mutex> lock(local_map_mutex_);
        
        if (timed_cloud_queue_.empty()) {
            RCLCPP_DEBUG(this->get_logger(), "No points to publish in local map");
            return;
        }
        
        // Create a fresh point cloud for the local map
        PointCloudXYZI::Ptr local_map(new PointCloudXYZI());
        
        // Add points from timed_cloud_queue_ with enhanced intensity based on recency
        for (size_t i = 0; i < timed_cloud_queue_.size(); i++) {
            // Calculate how recent this cloud is (0.0 = oldest, 1.0 = newest)
            double recency = static_cast<double>(i) / static_cast<double>(timed_cloud_queue_.size());
            
            // Get the points from this cloud
            const PointCloudXYZI::Ptr& cloud = timed_cloud_queue_[i].second;
            
            // Add these points to the local map with enhanced intensity based on recency
            for (const auto& point : cloud->points) {
                PointType new_point = point;
                
                // Scale intensity by recency (newer points are brighter)
                // Adjust these values to control the visual effect
                double time_factor = recency * 0.5 + 0.5; // ranges from 0.5 to 1.0
                new_point.intensity *= time_factor;
                
                local_map->points.push_back(new_point);
            }
        }
        
        // Ensure the width/height are set correctly for the point cloud
        local_map->width = local_map->points.size();
        local_map->height = 1;
        
        // Skip publishing if we have no points
        if (local_map->points.empty()) {
            RCLCPP_DEBUG(this->get_logger(), "No points to publish in local map after processing");
            return;
        }
        
        // Downsample the local map
        PointCloudXYZI::Ptr local_map_ds(new PointCloudXYZI());
        downSizeFilterMap.setInputCloud(local_map);
        downSizeFilterMap.filter(*local_map_ds);
        
        // Publish the local map
        sensor_msgs::msg::PointCloud2 local_map_msg;
        pcl::toROSMsg(*local_map_ds, local_map_msg);
        local_map_msg.header.stamp = this->get_clock()->now();
        local_map_msg.header.frame_id = "camera_init";
        pubLocalMap_->publish(local_map_msg);
        
        RCLCPP_INFO(this->get_logger(), "Published local map with %lu points from %lu clouds", 
                   local_map_ds->points.size(), timed_cloud_queue_.size());
    }
    
    // Add a method to accumulate points for the local map
    void accumulateLocalMap(const PointCloudXYZI::Ptr& transformed_cloud) {
        std::lock_guard<std::mutex> lock(local_map_mutex_);
        
        // Get current time
        rclcpp::Time current_time = this->get_clock()->now();
        
        // Make a copy of the transformed cloud to preserve original intensity values
        PointCloudXYZI::Ptr cloud_copy(new PointCloudXYZI(*transformed_cloud));
        
        // Add to our queue with timestamp
        timed_cloud_queue_.push_back(std::make_pair(current_time, cloud_copy));
        
        // Update last map time
        last_local_map_time_ = current_time;
        
        // In normal operation, use a shorter time window to show more dynamic changes
        // In relocalization mode, use a longer window for better matching
        double time_window = relocalization_mode ? 30.0 : 2.0; // seconds
        
        // Remove older clouds from the queue
        int removed_clouds = 0;
        while (!timed_cloud_queue_.empty()) {
            auto& oldest = timed_cloud_queue_.front();
            double age = (current_time - oldest.first).seconds();
            
            if (age > time_window) {
                timed_cloud_queue_.pop_front(); // Remove the oldest cloud
                removed_clouds++;
            } else {
                break; // Stop once we find a cloud that's within our time window
            }
        }
        
        // Log clouds kept and removed
        if (removed_clouds > 0) {
            RCLCPP_INFO(this->get_logger(), "Local map: Kept %lu recent clouds (%.1f sec window), removed %d old clouds", 
                      timed_cloud_queue_.size(), time_window, removed_clouds);
        }
    }

    // Publish a laser scan for costmap usage
    void publish_laser_scan(const pcl::PointCloud<PointType>::Ptr& cloud) {
        if (!cloud || cloud->empty()) {
            RCLCPP_WARN(this->get_logger(), "Empty point cloud, not publishing laser scan");
            return;
        }

        // Create a laser scan message
        sensor_msgs::msg::LaserScan laser_scan;
        laser_scan.header.stamp = this->now();
        laser_scan.header.frame_id = "camera_init"; // Match the frame used by FAST-LIO

        // Set the parameters for the laser scan
        const double angle_min = -M_PI;        // -180 degrees
        const double angle_max = M_PI;         // 180 degrees
        const int num_rays = 720;              // Number of rays in the scan (2 rays per degree)
        const double angle_increment = (angle_max - angle_min) / num_rays;

        // Configure laser scan
        laser_scan.angle_min = angle_min;
        laser_scan.angle_max = angle_max;
        laser_scan.angle_increment = angle_increment;
        laser_scan.time_increment = 0.0;
        laser_scan.scan_time = 0.1;            // 10Hz scan rate
        laser_scan.range_min = 0.3;            // Minimum range (matches blind parameter)
        laser_scan.range_max = 100.0;          // Maximum range

        // Initialize ranges to max range
        laser_scan.ranges.resize(num_rays, laser_scan.range_max);
        
        // Project 3D points into 2D laser scan
        for (const auto& point : cloud->points) {
            // Skip invalid points (NaN or Inf values)
            if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
                continue;
            }
            
            // Calculate the angle for this point
            double angle = std::atan2(point.y, point.x);
            
            // Normalize angle to range [angle_min, angle_max]
            while (angle < angle_min) angle += 2 * M_PI;
            while (angle > angle_max) angle -= 2 * M_PI;
            
            // Calculate the range
            double range = std::sqrt(point.x * point.x + point.y * point.y);
            
            // Skip points outside our desired range
            if (range < laser_scan.range_min || range > laser_scan.range_max) {
                continue;
            }
            
            // Calculate the index in the ranges array
            int index = static_cast<int>((angle - angle_min) / angle_increment);
            if (index >= 0 && index < num_rays) {
                // Only update if this point is closer than what we've seen so far
                if (range < laser_scan.ranges[index]) {
                    laser_scan.ranges[index] = range;
                }
            }
        }
        
        // Publish the laser scan
        pubLaserScan_->publish(laser_scan);
        RCLCPP_DEBUG(this->get_logger(), "Published laser scan with %d rays", num_rays);
    }

    // Filter points based on height thresholds for output only
    PointCloudXYZI::Ptr filterPointsByHeight(const PointCloudXYZI::Ptr& cloud_in) {
        PointCloudXYZI::Ptr cloud_filtered(new PointCloudXYZI());
        
        // Reserve memory for efficiency
        cloud_filtered->reserve(cloud_in->size());
        
        // Filter points based on height (z-coordinate)
        for (const auto& point : cloud_in->points) {
            if (point.z >= ground_level_ && point.z <= max_height_) {
                cloud_filtered->push_back(point);
            }
        }
        
        return cloud_filtered;
    }

private:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath_;
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr pubLaserScan_; // Added LaserScan publisher
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_pc_;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;

    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr map_save_srv_;

    bool effect_pub_en = false, map_pub_en = false;
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    double epsi[23] = {0.001};

    // Relocalization variables
    bool relocalization_done_ = false;
    bool relocalization_reset_needed_ = false;
    
    FILE *fp;
    ofstream fout_pre, fout_out, fout_dbg;
    
    // Loop closure related
    bool loop_closure_enabled_;
    double loop_closure_search_radius_;
    double loop_closure_min_distance_;
    int loop_closure_detection_interval_;
    double loop_closure_fitness_score_threshold_;
    std::vector<std::shared_ptr<KeyFrame>> keyframes_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pubLoopClosureMarker_;
    
    // Parameters for height-based point filtering
    double max_height_;
    double ground_level_;
    
    // Local map publishing
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLocalMap_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    signal(SIGINT, SigHandle);

    rclcpp::spin(std::make_shared<LaserMappingNode>());

    if (rclcpp::ok())
        rclcpp::shutdown();
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot10[i]), int(s_plot11[i]));
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
