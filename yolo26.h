#pragma once

#include <queue>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <mutex>
#include <cstdlib>

#define mxutil_prepost_sigmoid(_x_) (1.0 / (1.0 + expf(-1.0 * (_x_))))                                   // sigmoid: f(x) = 1 / (1 + e^(-x))
#define mxutil_prepost_sigmoid_fast_sigmoid(_x_) ((_x_) / (((_x_) < 0) ? (1.0 - (_x_)) : (1.0 + (_x_)))) // fast-sigmoid: f(x) = x / (1 + abs(x))
#define mxutil_max(_x_, _y_) (((_x_) > (_y_)) ? (_x_) : (_y_))
#define mxutil_min(_x_, _y_) (((_x_) < (_y_)) ? (_x_) : (_y_))

static const std::vector<cv::Scalar> COCO_TEXT_COLORS = {
    {0, 0, 0},
    {255, 255, 255},
    {255, 255, 255},
    {255, 255, 255},
    {255, 215, 0},
};

static const std::vector<cv::Scalar> COCO_BOX_COLORS = {
    {255, 255, 0, 0.6},
    {26, 35, 126, 0.6},
    {255, 50, 50, 0.6},
    {0, 0, 0, 0.6},
    {51, 51, 51, 0.6},
};

struct BBox
{
    int class_index;   // class index with maximum confident
    float class_score; // class confident(score)
    float x_min;       // global top-left x relates to model's input feature map size width
    float y_min;       // global top-left y relates to model's input feature map size height
    float x_max;       // global bottom-right x relates to model's input feature map size width
    float y_max;       // global bottom-right y relates to model's input feature map size height

    // Default constructor
    BBox() : class_index(-1), class_score(-1), x_min(-1), y_min(-1), x_max(-1), y_max(-1) {}
    // Parameterized constructor
    BBox(int _class_index, float _class_socre, float _x_min, float _y_min, float _x_max, float _y_max)
        : class_index(_class_index), class_score(_class_socre), x_min(_x_min), y_min(_y_min), x_max(_x_max), y_max(_y_max) {}
};

struct YOLO26Result
{
    std::queue<BBox> bboxes;
    std::queue<std::vector<std::pair<float, float>>> keypoints;
    std::queue<std::vector<float>> mask_features;
    std::queue<cv::Rect> final_rois;
    std::queue<cv::Mat> final_masks;
};

class YOLO26
{
public:
    /** @brief Constructor for using official 80 classes COCO dataset. */
    YOLO26();

    cv::Mat preprocess(const cv::Mat& image);

    /**
     * @brief Post-process the output data from the YOLO26 model.
     * @param output_buffers   Vector of pointers to the output buffers from the accelerator.
     * @param result           Reference to the structure where the decoded bounding box results will be stored.
     */
    void postprocess(std::vector<float *> output_buffers, YOLO26Result &result);

    /** @brief Draw detected bounding boxes and labels on the provided image. */
    void draw_result(YOLO26Result &result, cv::Mat &image);

    /** @brief Compute padding values for letterboxing from the display image. */
    void compute_padding(int disp_width, int disp_height);

    /** @brief Ensure the input dimensions are valid for horizontal display images only.  */
    bool is_horizontal_input(int disp_width, int disp_height);

    /**
     * @brief Calculate the area overlap between two bounding box for NMS algo. to
     * combine or drop bounding boxes.
     *
     * @param bbox_0            bounding box 0
     * @param bbox_1            bounding box 1
     * @param class_chk         set zero to ignore class type
     *
     * @return overlap percentage
     */
    float intersection_over_union(BBox &bbox_0, BBox &bbox_1, int class_chk);

    /**
     * @brief Post-process to calculate detection overlaps to combine the same
     * object with duplicated detections together as only one detection. Bounding
     * box will be added to list if IOU is smaller then given threshold, otherwise
     * the bounding box with higher score will be kept.
     *
     * @param bboxes_detected   linked-list which stores bounding boxes detected
     * @param bbox              bounding box to be added to list
     * @param iou               IOU threshold to combine bounding box
     *
     * @return none
     */
    void non_maximum_suppression(std::queue<BBox> &bboxes, BBox &bbox, float iou);

private:
    /** @brief Structure representing per-layer information of YOLO26 output. */
    struct LayerParams
    {
        uint8_t coordinate_ofmap_flow_id;
        uint8_t confidence_ofmap_flow_id;
        size_t width;
        size_t height;
        size_t ratio;
        size_t coordinate_fmap_size;
    };

    /** @brief Initialization method to set up YOLO26 model parameters. */
    void Init(size_t model_input_width, size_t model_input_height, size_t model_input_channel,
              float confidence_thresh, float iou_thresh, size_t class_count, const char **class_labels);

    /** @brief Helper methods for building detections from model output. */
    void _get_detection(std::queue<BBox> &bounding_boxes, int layer_id, float *confidence_buffer,
                        float *coordinate_buffer, int row, int col, float *confs_tmp);
    
    void _draw_bbox(cv::Mat &image, int x_min, int y_min, int x_max, int y_max,
                    cv::Scalar box_color, cv::Scalar text_color, const char *class_name, const float class_score);
    
    float _conf_to_fastSigmoid_inputVal(float conf);
    
    static constexpr size_t kNumPostProcessLayers = 3;
    struct LayerParams yolo_post_layers_[kNumPostProcessLayers];

    // Model-specific parameters.
    const char **class_labels_;
    size_t class_count_;
    size_t model_input_width_;   // Input width to accelerator, obtained by dfp.
    size_t model_input_height_;  // Input height to accelerator, obtained by dfp.
    size_t model_input_channel_; // Input channel to accelerator, obtained by dfp.

    // Colors for labels and bounding boxes.
    std::vector<cv::Scalar> class_label_colors_;
    std::vector<cv::Scalar> bounding_box_colors_;

    // Confidence and IOU thresholds.
    std::mutex confidence_mutex_;
    float confidence_thresh_;
    float confidence_thresh_fastSigmoid_; // Converted confidence threshold for fast-sigmoid
    float iou_thresh_;

    // Letterbox ratio and padding.
    float letterbox_ratio_;
    int letterbox_width_;
    int letterbox_height_;
    int padding_height_;
    int padding_width_;
    bool valid_input_;
};
