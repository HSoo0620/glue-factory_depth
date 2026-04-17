/*
 * SHOT352 Descriptor - pybind11 module (cross-platform)
 * Input:  numpy array (N,3) float32 point cloud (mm)
 * Output: dict { "points": (M,3), "descriptors": (M,352), stats... }
 *
 * No pcl_io link -> No OpenNI2 dependency
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>

#include <cmath>
#include <iostream>

namespace py = pybind11;

py::dict extract_shot(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_np,
    float voxel_size,
    float normal_radius,
    float shot_radius)
{
    auto buf = points_np.unchecked<2>();
    if (buf.shape(1) != 3)
        throw std::runtime_error("points must be (N, 3)");

    size_t n = buf.shape(0);

    // 1. Numpy -> PCL PointCloud
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(n);
    for (size_t i = 0; i < n; i++) {
        cloud->push_back(pcl::PointXYZ(buf(i, 0), buf(i, 1), buf(i, 2)));
    }
    cloud->width  = static_cast<uint32_t>(n);
    cloud->height = 1;
    cloud->is_dense = true;

    std::cout << "  Input points: " << cloud->size() << std::endl;

    // 2. VoxelGrid Downsampling
    auto downsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*downsampled);

    std::cout << "  Downsampled: " << cloud->size() << " -> " << downsampled->size()
              << " (voxel=" << voxel_size << "mm)" << std::endl;

    if (downsampled->size() < 10)
        throw std::runtime_error("Too few points after downsampling");

    // 3. Normal Estimation (OMP)
    auto normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(downsampled);
    auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(normal_radius);
    ne.compute(*normals);

    std::cout << "  Normals computed (radius=" << normal_radius << "mm)" << std::endl;

    // Remove NaN normals
    auto cleanCloud   = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto cleanNormals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    for (size_t i = 0; i < normals->size(); i++) {
        if (std::isfinite(normals->at(i).normal_x) &&
            std::isfinite(normals->at(i).normal_y) &&
            std::isfinite(normals->at(i).normal_z)) {
            cleanCloud->push_back(downsampled->at(i));
            cleanNormals->push_back(normals->at(i));
        }
    }
    cleanCloud->width    = static_cast<uint32_t>(cleanCloud->size());
    cleanCloud->height   = 1;
    cleanCloud->is_dense = true;
    cleanNormals->width  = static_cast<uint32_t>(cleanNormals->size());
    cleanNormals->height = 1;
    cleanNormals->is_dense = true;

    std::cout << "  Valid normals: " << cleanCloud->size() << " / "
              << downsampled->size() << std::endl;

    if (cleanCloud->size() < 10)
        throw std::runtime_error("Too few valid normals");

    // 4. SHOT352 Descriptor
    auto shotTree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    shot.setInputCloud(cleanCloud);
    shot.setInputNormals(cleanNormals);
    shot.setSearchMethod(shotTree);
    shot.setRadiusSearch(shot_radius);

    auto descriptors = pcl::make_shared<pcl::PointCloud<pcl::SHOT352>>();
    shot.compute(*descriptors);

    int validDesc = 0;
    for (size_t i = 0; i < descriptors->size(); i++) {
        if (std::isfinite(descriptors->at(i).descriptor[0])) validDesc++;
    }
    std::cout << "  SHOT352: " << validDesc << " / " << descriptors->size()
              << " valid (radius=" << shot_radius << "mm)" << std::endl;

    // 5. Build numpy arrays
    size_t m = cleanCloud->size();

    py::array_t<float> out_points({m, (size_t)3});
    py::array_t<float> out_desc({m, (size_t)352});
    auto pts_mut  = out_points.mutable_unchecked<2>();
    auto desc_mut = out_desc.mutable_unchecked<2>();

    for (size_t i = 0; i < m; i++) {
        pts_mut(i, 0) = cleanCloud->at(i).x;
        pts_mut(i, 1) = cleanCloud->at(i).y;
        pts_mut(i, 2) = cleanCloud->at(i).z;
        for (int d = 0; d < 352; d++) {
            desc_mut(i, d) = descriptors->at(i).descriptor[d];
        }
    }

    py::dict result;
    result["points"]          = out_points;
    result["descriptors"]     = out_desc;
    result["num_input"]       = static_cast<int>(n);
    result["num_downsampled"] = static_cast<int>(downsampled->size());
    result["num_keypoints"]   = static_cast<int>(m);
    result["num_valid_desc"]  = validDesc;

    return result;
}

py::dict extract_shot_at_keypoints(
    py::array_t<float, py::array::c_style | py::array::forcecast> points_np,
    py::array_t<float, py::array::c_style | py::array::forcecast> keypoints_np,
    float voxel_size,
    float normal_radius,
    float shot_radius)
{
    auto buf = points_np.unchecked<2>();
    auto kbuf = keypoints_np.unchecked<2>();
    if (buf.shape(1) != 3) throw std::runtime_error("points must be (N, 3)");
    if (kbuf.shape(1) != 3) throw std::runtime_error("keypoints must be (K, 3)");

    size_t n = buf.shape(0);
    size_t k = kbuf.shape(0);

    // 1. Numpy -> PCL clouds
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->reserve(n);
    for (size_t i = 0; i < n; i++)
        cloud->push_back(pcl::PointXYZ(buf(i, 0), buf(i, 1), buf(i, 2)));
    cloud->width = static_cast<uint32_t>(n); cloud->height = 1; cloud->is_dense = true;

    auto kp_cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    kp_cloud->reserve(k);
    for (size_t i = 0; i < k; i++)
        kp_cloud->push_back(pcl::PointXYZ(kbuf(i, 0), kbuf(i, 1), kbuf(i, 2)));
    kp_cloud->width = static_cast<uint32_t>(k); kp_cloud->height = 1; kp_cloud->is_dense = true;

    std::cout << "  Input points: " << cloud->size()
              << ", keypoints: " << kp_cloud->size() << std::endl;

    // 2. VoxelGrid downsample (search surface)
    auto downsampled = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(voxel_size, voxel_size, voxel_size);
    vg.filter(*downsampled);
    std::cout << "  Voxel: " << cloud->size() << " -> " << downsampled->size()
              << " (voxel=" << voxel_size << "mm)" << std::endl;
    if (downsampled->size() < 10) throw std::runtime_error("Too few voxel points");

    // 3. Normal estimation on voxel cloud
    auto normals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(downsampled);
    auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(normal_radius);
    ne.compute(*normals);

    // 4. Remove NaN normals
    auto cleanCloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto cleanNormals = pcl::make_shared<pcl::PointCloud<pcl::Normal>>();
    for (size_t i = 0; i < normals->size(); i++) {
        if (std::isfinite(normals->at(i).normal_x) &&
            std::isfinite(normals->at(i).normal_y) &&
            std::isfinite(normals->at(i).normal_z)) {
            cleanCloud->push_back(downsampled->at(i));
            cleanNormals->push_back(normals->at(i));
        }
    }
    cleanCloud->width = (uint32_t)cleanCloud->size(); cleanCloud->height = 1; cleanCloud->is_dense = true;
    cleanNormals->width = (uint32_t)cleanNormals->size(); cleanNormals->height = 1; cleanNormals->is_dense = true;
    if (cleanCloud->size() < 10) throw std::runtime_error("Too few valid normals");

    // 5. SHOT352 at keypoints (search surface = cleanCloud)
    auto shotTree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
    shot.setSearchMethod(shotTree);
    shot.setSearchSurface(cleanCloud);     // 이웃 검색 surface
    shot.setInputNormals(cleanNormals);
    shot.setInputCloud(kp_cloud);          // 결과는 keypoint 위치에서만
    shot.setRadiusSearch(shot_radius);

    auto descriptors = pcl::make_shared<pcl::PointCloud<pcl::SHOT352>>();
    shot.compute(*descriptors);

    if (descriptors->size() != k)
        throw std::runtime_error("SHOT descriptor count mismatch with keypoint count");

    // 6. NaN descriptor -> 0 vector, build valid_mask
    py::array_t<float> out_desc({k, (size_t)352});
    py::array_t<bool> out_valid(k);
    auto desc_mut = out_desc.mutable_unchecked<2>();
    auto valid_mut = out_valid.mutable_unchecked<1>();
    int validDesc = 0;
    for (size_t i = 0; i < k; i++) {
        bool is_valid = std::isfinite(descriptors->at(i).descriptor[0]);
        valid_mut(i) = is_valid;
        if (is_valid) validDesc++;
        for (int d = 0; d < 352; d++) {
            float v = descriptors->at(i).descriptor[d];
            desc_mut(i, d) = (is_valid && std::isfinite(v)) ? v : 0.0f;
        }
    }

    py::dict result;
    result["descriptors"]    = out_desc;
    result["valid_mask"]     = out_valid;
    result["num_input"]      = static_cast<int>(n);
    result["num_voxel"]      = static_cast<int>(downsampled->size());
    result["num_keypoints"]  = static_cast<int>(k);
    result["num_valid_desc"] = validDesc;
    return result;
}

PYBIND11_MODULE(shot_module, m) {
    m.doc() = "SHOT352 descriptor extractor using PCL (pybind11)";
    m.def("extract_shot", &extract_shot,
          "Extract SHOT352 descriptors from point cloud (N,3) float32 in mm",
          py::arg("points"),
          py::arg("voxel_size")    = 5.0f,
          py::arg("normal_radius") = 25.0f,
          py::arg("shot_radius")   = 50.0f);
    m.def("extract_shot_at_keypoints", &extract_shot_at_keypoints,
          "Extract SHOT352 descriptors only at given keypoint XYZ positions.\n"
          "Returns dict{descriptors(K,352) float32, valid_mask(K,) bool, "
          "num_input, num_voxel, num_keypoints, num_valid_desc}.\n"
          "Contract: when valid_mask[i]==False the corresponding descriptors row "
          "is zero-filled (callers must handle zero rows to avoid division-by-zero "
          "during L2 normalisation).",
          py::arg("points"),
          py::arg("keypoints"),
          py::arg("voxel_size")    = 1.0f,
          py::arg("normal_radius") = 20.0f,
          py::arg("shot_radius")   = 40.0f);
}
