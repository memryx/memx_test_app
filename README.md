# C++ Object Detection with YOLO26 for Renesas RZ/G3E

The **Object Detection** example demonstrates real-time object detection on one or more input streams using the pre-trained YOLO26 nano model on MemryX accelerators. This guide provides application and model details, and necessary code snippets to help you quickly get started. For building and deploying on the RZ/G3E, please refer to the instructions in the [memx-yocto-renesas](https://github.com/memryx/memx-yocto-renesas) repository.

<p align="center">
  <img src="assets/objectDetection_yolo26n.png" alt="Object Detection Example" width="45%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolo26n](https://docs.ultralytics.com/models/yolo26/)                                            |
| **Model Type**       | Object Detection                                                      |
| **Framework**        | [onnx](https://onnx.ai/)                                                   |
| **Model Source**     | [Download from Ultralytics GitHub or docs](https://docs.ultralytics.com/tasks/detect/) |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/model_explorer/2p2/YOLO26_nano_640_640_3_onnx.zip)       |
| **Dataset**          | [COCO](https://docs.ultralytics.com/datasets/detect/coco/) |
| **Model Resolution**            | 640x640                                                   |
| **Output**           | Bounding box coordinates with object probabilities |
| **OS**           | Linux |
| **License**          | [AGPL](LICENSE.md)                                      |

## Command Line Arguments

The application supports the following command-line options:

| Argument | Description | Default |
|----------|-------------|---------|
| `-d, --dfp_path` | Path to the DFP (Deep Fusion Package) model file | `../../assets/models/YOLO26_nano_640_640_3_onnx.dfp` |
| `--video_paths` | Video source paths in the format `"cam:0,vid:video_path,vid:video2_path"`. Use `cam:N` for camera device N, or `vid:path` for video files. Multiple sources can be specified separated by commas. | `vid:../../assets/video/sample.mp4` |
| `--show` | Display the inference results in a window | `false` |

**Example usage:**
```bash
# Run with default settings 
./main

# Run with camera input and display
./main --video_paths "cam:0" --show

# Run with custom model and video file
./main -d path/to/model.dfp --video_paths "vid:path/to/video.mp4"

# Run with multiple video sources
./main --video_paths "cam:0,vid:video1.mp4,vid:video2.mp4" --show
```

