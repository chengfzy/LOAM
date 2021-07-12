// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk

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

/**
 * @note NOTE，注意代码中的坐标系定义
 *
 * velodyne LiDAR被安装为[X,Y,Z]=[前,左,上]
 *
 */

#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <cmath>
#include <string>
#include <vector>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"

using std::atan2;
using std::cos;
using std::sin;

// 扫描周期, velodyne频率10Hz, 周期0.1s
const double scanPeriod = 0.1;

// 初始化控制变量
const int systemDelay = 0;  // 弃用前N帧初始化数据
int systemInitCount = 0;    // 用来统计当前已经弃用的帧数
bool systemInited = false;  // 是否已经初始化

int N_SCANS = 0;  // 激光雷达线数

float cloudCurvature[400000];     //点云曲率, 40000为一帧点云中点的最大数量
int cloudSortInd[400000];         //曲率对应的点的index
int cloudNeighborPicked[400000];  //点是否被筛选过的标志, 0:筛选过, 1: 筛选过
//点分类标号, 2: 曲率很大,  1: 曲率比较大, 0: 曲率比较小, -1: 曲率很小. 其中1包括2, 0包括1. 0和1构成了点云全部的点
int cloudLabel[400000];

bool comp(int i, int j) { return (cloudCurvature[i] < cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1;

/**
 * @brief 去掉点云中比较近的点
 *
 * @tparam PointT   点云类型
 * @param cloud_in  输入点云
 * @param cloud_out 输出点云
 * @param thresh    阈值, 小于该阈值中的点将被删除
 */
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT>& cloud_in, pcl::PointCloud<PointT>& cloud_out, float thresh) {
    if (&cloud_in != &cloud_out) {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
    for (size_t i = 0; i < cloud_in.points.size(); ++i) {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y +
                cloud_in.points[i].z * cloud_in.points[i].z <
            thresh * thresh) {
            continue;
        }
        cloud_out.points[j] = cloud_in.points[i];
        ++j;
    }
    if (j != cloud_in.points.size()) {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

/**
 * @brief 输入点云消息的处理函数
 *
 * @param laserCloudMsg 点云消息
 *
 * @note 关于点云这块的处理, 参考https://blog.csdn.net/baidu_34319491/article/details/117396349
 */
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) {
    // 初始化
    if (!systemInited) {
        systemInitCount++;
        if (systemInitCount >= systemDelay) {
            systemInited = true;
        } else {
            return;
        }
    }

    TicToc t_whole;
    TicToc t_prepare;
    // 记录每个scan的开始和结束Index
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    // 消息转换成PCL数据
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;

    // 移除掉空点和比较近的点
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    int cloudSize = laserCloudIn.points.size();
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

    // 结束方位角与开始方位角差控制在[PI, 3*PI]范围, 允许LiDAR不是一个圆周扫描. 正常情况下在这个范围内, 异常则修正.
    // 更重要的是保证起始和终止帧在2*PI附近, 参考https://blog.csdn.net/baidu_34319491/article/details/117396349
    if (endOri - startOri > 3 * M_PI) {
        endOri -= 2 * M_PI;
    } else if (endOri - startOri < M_PI) {
        endOri += 2 * M_PI;
    }
    // printf("end Ori %f\n", endOri);

    bool halfPassed = false;  // 扫描是否旋转过半
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);  // 按scan进行分组
    for (int i = 0; i < cloudSize; i++) {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;

        // 计算仰角, 并根据仰角排列激光线号, velodyne每两个scan之间间隔2度
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;
        if (N_SCANS == 16) {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0) {
                count--;
                continue;
            }
        } else if (N_SCANS == 32) {
            scanID = int((angle + 92.0 / 3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0) {
                count--;
                continue;
            }
        } else if (N_SCANS == 64) {
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0) {
                count--;
                continue;
            }
        } else {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        // printf("angle %f scanID %d \n", angle, scanID);

        // 根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算, 从而进行补偿
        float ori = -atan2(point.y, point.x);  // 旋转角
        if (!halfPassed) {
            // 确保-PI/2 < ori - startOri < 3/2*PI
            if (ori < startOri - M_PI / 2) {
                ori += 2 * M_PI;
            } else if (ori > startOri + M_PI * 3 / 2) {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI) {
                halfPassed = true;
            }
        } else {
            // 确保-PI/2 < endOri - ori < 3/2*PI
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2) {
                ori += 2 * M_PI;
            } else if (ori > endOri + M_PI / 2) {
                ori -= 2 * M_PI;
            }
        }

        // 点的强度=线号+相对时间, 即整数+小数, 整数部分表示线号, 小数部分表示该点的相对时间
        float relTime = (ori - startOri) / (endOri - startOri);
        point.intensity = scanID + scanPeriod * relTime;
        laserCloudScans[scanID].push_back(point);
    }
    cloudSize = count;
    printf("points size %d \n", cloudSize);

    // 将所有的点按线号(scan)从小到大放入一个容器, 并记录其开始和结束的index(去掉首尾5个点)
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++) {
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];  //只是指向该线号的点云数据指针
        scanEndInd[i] = laserCloud->size() - 6;
    }
    printf("prepare time %f \n", t_prepare.toc());

    // 使用每个点的前后5个点来计算曲率, 可以看成是中心差分的扩展形式, d[n] = (f[n-1] + f[n] - 2*f[n]) / 2,
    // 需要注意除数2(这里是10)省掉了.
    for (int i = 5; i < cloudSize - 5; i++) {
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x +
                      laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x +
                      laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x +
                      laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y +
                      laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y +
                      laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y +
                      laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z +
                      laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z +
                      laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z +
                      laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;  // 曲率
        cloudSortInd[i] = i;                                                // 记录曲率对应的点的index
        cloudNeighborPicked[i] = 0;                                         // 初始化时, 点全未筛选过
        cloudLabel[i] = 0;                                                  // 初始化为less flat点(曲率较小)
    }

    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;      // 曲率较大, 对应边缘
    pcl::PointCloud<PointType> cornerPointsLessSharp;  // 曲率比较大
    pcl::PointCloud<PointType> surfPointsFlat;         // 曲率较小, 对应平面
    pcl::PointCloud<PointType> surfPointsLessFlat;     // 曲率比较小

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++) {
        // 保证每个线号至少有6个点. TODO: 什么时候会出现这种情况?
        if (scanEndInd[i] - scanStartInd[i] < 6) {
            continue;
        }
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        // 将每个scan的曲率点分成6等份处理, 确保周围都有点被选作特征点
        for (int j = 0; j < 6; j++) {
            // start patch, 六等份的起点, sp = scanStartInd + (scanEndInd - scanStartInd) * j / 6
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
            // end patch, 六等份的终点, ep = scanStartInd + (scanEndInd - scanStartInd) * (j + 1) / 6 - 1
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            // 对每个分段, 将曲率按从小到大排列
            TicToc t_tmp;
            std::sort(cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            t_q_sort += t_tmp.toc();

            // 挑选每个分段中曲率很大和比较大的点
            int largestPickedNum = 0;
            // 因为曲率是按照从小到大排列, 所以这里逆序处理
            for (int k = ep; k >= sp; k--) {
                int ind = cloudSortInd[k];  // 点对应的index

                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1) {
                    largestPickedNum++;
                    if (largestPickedNum <= 2) {  // 曲率最大的前2个点放入sharp点集合
                        cloudLabel[ind] = 2;      // 2代表曲率很大
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    } else if (largestPickedNum <= 20) {  // 将曲率最大的前20个点放入less sharp点集合
                        cloudLabel[ind] = 1;              // 1代表曲率比较大
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    } else {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;  // 筛选标志置位, 表示点被筛选过了

                    // 将曲率比较大的点前后5个连续的点筛选出去, 防止特征点聚集, 使得特征点在各个方向上尽量分布比较均匀.
                    for (int l = 1; l <= 5; l++) {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--) {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 挑选每个分段中曲率很小和比较小的点
            int smallestPickedNum = 0;
            // 因为曲率是按照从小到大排列, 所以这里顺序处理
            for (int k = sp; k <= ep; k++) {
                int ind = cloudSortInd[k];  // 点对应的index

                if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1) {
                    cloudLabel[ind] = -1;  // -1代码曲率很小
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4) {  // 只选最小的4个, 剩下的的label为0, 均是曲率比较小的点
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    // 同样防止特征聚集
                    for (int l = 1; l <= 5; l++) {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--) {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 将剩余的点, 包括之前被排除的点, 全部归入平面点中less flat类别中
            for (int k = sp; k <= ep; k++) {
                if (cloudLabel[k] <= 0) {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        // 由于less flat点最多, 对其进行体素栅格滤波
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        // less flat点汇总
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }

    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlatMsg;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlatMsg);
    surfPointsFlatMsg.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlatMsg.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlatMsg);

    sensor_msgs::PointCloud2 surfPointsLessFlatMsg;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlatMsg);
    surfPointsLessFlatMsg.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlatMsg.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlatMsg);

    // pub each scam
    if (PUB_EACH_LINE) {
        for (int i = 0; i < N_SCANS; i++) {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if (t_whole.toc() > 100) ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 16);

    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.3);

    printf("scan line number %d \n", N_SCANS);

    if (N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64) {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if (PUB_EACH_LINE) {
        for (int i = 0; i < N_SCANS; i++) {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
