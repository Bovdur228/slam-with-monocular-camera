# Monocular SLAM from Scratch

Educational implementation of a monocular SLAM system in Python using OpenCV.

## Project Overview

This project aims to build a complete monocular SLAM pipeline from scratch for educational and research purposes.

The current implementation performs:

* Camera calibration
* ORB feature extraction
* Feature matching
* Essential matrix estimation
* Camera pose recovery
* Triangulation of 3D map points
* Global camera trajectory estimation
* KeyFrame creation
* MapPoint management
* Reprojection error evaluation

The project is being developed incrementally with the goal of implementing the major components used in modern SLAM systems.

---

## Current Pipeline

1. Capture video from a monocular camera.
2. Undistort frames using calibration parameters.
3. Extract ORB features.
4. Match features between consecutive frames.
5. Estimate the Essential Matrix using RANSAC.
6. Recover relative camera pose.
7. Triangulate new 3D points.
8. Convert points into world coordinates.
9. Manage MapPoints and KeyFrames.
10. Estimate reprojection error.

---

## Project Structure

```text
SLAM_monocular_cam/

├── capture_calibration_images.py
├── mono_calibration.py
├── monocular_SLAM.py
```

### Files

* `capture_calibration_images.py` — captures calibration images.
* `mono_calibration.py` — computes camera intrinsics and distortion coefficients.
* `monocular_SLAM.py` — main SLAM pipeline.

---

## Planned Features

* Local Bundle Adjustment
* KeyFrame culling
* MapPoint culling
* PnP tracking against the map
* Loop Closure
* 3D Point Cloud view
* Pose Graph Optimization
* Better MapPoint matching strategy

---

## Technologies

* Python
* OpenCV
* NumPy

---

## Status

Work in progress.
This repository is intended primarily as a learning and experimentation project focused on understanding the internal architecture of modern SLAM systems.
