#include "yolo26.h"

#define FONT (cv::FONT_ITALIC)
#define COCO_CLASS_NUMBER (80)

/**
 * @brief Labels of COCO dataset, COCO 2014 and 2017 uses the same images but
 * different train/val/test splits. Also, COCO defines 91 classes but the data
 * only uses 80 classes.
 */
const char *COCO_NAMES[COCO_CLASS_NUMBER] = {
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
};

float YOLO26::intersection_over_union(BBox &bbox_0, BBox &bbox_1, int class_chk)
{
    if (class_chk)
    {
        if (bbox_0.class_index != bbox_1.class_index)
            return 0.0;
    }

    float y_min = mxutil_max(bbox_0.y_min, bbox_1.y_min);
    float x_min = mxutil_max(bbox_0.x_min, bbox_1.x_min);
    float y_max = mxutil_min(bbox_0.y_max, bbox_1.y_max);
    float x_max = mxutil_min(bbox_0.x_max, bbox_1.x_max);
    float intersection_area = mxutil_max(0, (y_max - y_min)) * mxutil_max(0, (x_max - x_min));
    float bbox_0_area = (bbox_0.y_max - bbox_0.y_min) * (bbox_0.x_max - bbox_0.x_min);
    float bbox_1_area = (bbox_1.y_max - bbox_1.y_min) * (bbox_1.x_max - bbox_1.x_min);
    float union_area = bbox_0_area + bbox_1_area - intersection_area;
    return intersection_area / union_area;
}

void YOLO26::non_maximum_suppression(std::queue<BBox> &bboxes, BBox &bbox, float iou)
{
    BBox bbox_0;
    BBox bbox_to_be_stored;
    int count = bboxes.size();

    int exit_flag = 0;

    // iterative over all bounding boxes
    for (int i = 0; i < count; ++i)
    {
        bbox_0 = bboxes.front();
        bboxes.pop();
        if (intersection_over_union(bbox_0, bbox, 0) > iou)
        {
            // if two bounding boxes are highly overlapped, keep the one with higher score
            if (bbox_0.class_score > bbox.class_score)
            {
                bbox_to_be_stored = bbox_0;
                exit_flag = 1;
            }
            else
            {
                bbox_to_be_stored = bbox;
            }
            if (exit_flag)
            {
                bboxes.push(bbox_to_be_stored);
                return;
            }
            else
            {
                if (i == count - 1)
                { // end of comparison
                    bboxes.push(bbox_to_be_stored);
                    return;
                }
            }
        }
        else
        {
            // otherwise, put it back to list
            bboxes.push(bbox_0);
        }
    }
    // if not return from loop, we then add the given bounding box to list
    bboxes.push(bbox);
}

void YOLO26::_draw_bbox(cv::Mat &image, int x_min, int y_min, int x_max,
                        int y_max, cv::Scalar box_color, cv::Scalar text_color,
                        const char *class_name, const float class_score) 
{
    double font_scale = ((double)image.rows / 640.0);
    double bbox_thickness = font_scale * 3;
    double font_thickness = font_scale * 2;
    cv::Size text_size;
    int baseline;
    char text[64];
    /* bounding box rectangle line */
    cv::rectangle(image, cv::Point(x_min, y_min) /*top left*/, cv::Point(x_max, y_max) /*bottom right*/,
                  box_color, bbox_thickness, cv::LINE_4);

    sprintf(text, "%s(%.f%%)", class_name, 100 * class_score);

    text_size = cv::getTextSize(text, FONT, 2 * font_scale, bbox_thickness, &baseline);

    /* label background rectangle */
    cv::rectangle(image,
                  cv::Rect(x_min, mxutil_max(0, y_min - text_size.height), text_size.width * 0.5, text_size.height), // top left, width, height
                  box_color, cv::FILLED);

    /* label text */
    cv::putText(image, text,
                cv::Point(x_min, mxutil_max(0, y_min - text_size.height) == 0 ? text_size.height - 5 : y_min - 10 * font_scale), // bottom left
                FONT, font_scale, text_color, font_thickness, cv::LINE_AA);
}

YOLO26::YOLO26()
{
    model_input_width_ = 640;
    model_input_height_ = 640;
    model_input_channel_ = 3;
    confidence_thresh_ = 0.3f;
    confidence_thresh_fastSigmoid_ = _conf_to_fastSigmoid_inputVal(confidence_thresh_);
    iou_thresh_ = 0.45f;
    class_labels_ = COCO_NAMES;
    class_count_ = COCO_CLASS_NUMBER;
    int color_size = COCO_TEXT_COLORS.size();
    valid_input_ = false;

    // set label and bbox color
    for (size_t i = 0; i < class_count_; i++)
    {
        cv::Scalar label_color = COCO_TEXT_COLORS[i % color_size];
        cv::Scalar bbox_color = COCO_BOX_COLORS[i % color_size];
        class_label_colors_.push_back(label_color);
        bounding_box_colors_.push_back(bbox_color);
    }

    // YOLO26 (YOLOv11) uses direct bbox regression (4 values) instead of DFL (64 values)
    yolo_post_layers_[0] =
        {
            .coordinate_ofmap_flow_id = 0,
            .confidence_ofmap_flow_id = 1,
            .width = model_input_width_ / 8,   // L0_HW, 640 / 8 = 80
            .height = model_input_height_ / 8, // L0_HW, 640 / 8 = 80
            .ratio = 8,
            .coordinate_fmap_size = 4,  // YOLO26: 4 bbox values instead of 64
        };

    yolo_post_layers_[1] =
        {
            .coordinate_ofmap_flow_id = 2,
            .confidence_ofmap_flow_id = 3,
            .width = model_input_width_ / 16,   // L1_HW, 640 / 16 = 40
            .height = model_input_height_ / 16, // L1_HW, 640 / 16 = 40
            .ratio = 16,
            .coordinate_fmap_size = 4,  // YOLO26: 4 bbox values instead of 64
        };

    yolo_post_layers_[2] =
        {
            .coordinate_ofmap_flow_id = 4,
            .confidence_ofmap_flow_id = 5,
            .width = model_input_width_ / 32,   // L2_HW, 640 / 32 = 20
            .height = model_input_height_ / 32, // L2_HW, 640 / 32 = 20
            .ratio = 32,
            .coordinate_fmap_size = 4,  // YOLO26: 4 bbox values instead of 64
        };
}

bool YOLO26::is_horizontal_input(int disp_width, int disp_height)
{
    if (disp_height > disp_width)
    {
        printf("Invalid display image: only horizontal images are supported.\n");
        return false;
    }
    return true;
}

void YOLO26::compute_padding(int disp_width, int disp_height)
{
    if (!is_horizontal_input(disp_width, disp_height))
        return;

    letterbox_ratio_ = (float)model_input_width_ / disp_width;

    letterbox_width_ = disp_width * letterbox_ratio_;
    letterbox_height_ = disp_height * letterbox_ratio_;

    padding_width_ = (model_input_width_ - letterbox_width_) / 2;
    padding_height_ = (model_input_height_ - letterbox_height_) / 2;

    valid_input_ = true;
}

cv::Mat YOLO26::preprocess(const cv::Mat &image) {

    // Convert BGR to RGB (camera/video input is BGR, model expects RGB)
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);

    // Resize keeping aspect ratio
    cv::Mat resized;
    cv::resize(rgbImage, resized, cv::Size(letterbox_width_, letterbox_height_), 0,
               0, cv::INTER_LINEAR);

    // Apply letterbox padding (black border)
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       padding_height_, // top,
                       padding_height_, // bottom
                       padding_width_,  // left
                       padding_width_,  // right
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // Convert to float and normalize (0–1)
    padded.convertTo(padded, CV_32F, 1.0 / 255.0);
    return padded; // shape: (640, 640, 3), range [0,1]
}

void YOLO26::draw_result(YOLO26Result &result, cv::Mat &image)
{
    static int y_min, x_min, y_max, x_max, class_index;
    float class_score;
    std::queue<BBox> bboxes = result.bboxes;

    while (!bboxes.empty())
    {
        BBox bbox = bboxes.front();
        bboxes.pop();

        class_index = bbox.class_index;
        class_score = bbox.class_score;

        x_min = static_cast<int>((bbox.x_min - padding_width_) / letterbox_ratio_);
        y_min = static_cast<int>((bbox.y_min - padding_height_) / letterbox_ratio_);
        x_max = static_cast<int>((bbox.x_max - padding_width_) / letterbox_ratio_);
        y_max = static_cast<int>((bbox.y_max - padding_height_) / letterbox_ratio_);

        _draw_bbox(image, x_min, y_min, x_max, y_max,
                   bounding_box_colors_[class_index],
                   class_label_colors_[class_index],
                   class_labels_[class_index], class_score);
    }

    return;
}

float YOLO26::_conf_to_fastSigmoid_inputVal(float conf) {
    // Converts a confidence value in [0,1] to the corresponding input for the fast-sigmoid function.
    // The fast-sigmoid function: f(x) = x / (1 + |x|), which maps [-inf, +inf] -> [-1, 1].
    // Steps:
    //   1. Map conf [0,1] -> x [-1,1]
    //   2. Return x as input for fast-sigmoid
    
    float x = conf * 2.0f - 1.0f;  // map [0,1] -> [-1,1]
    return x / (1.0f - std::abs(x)); // map [-1, 1] -> [-inf, inf]
}

void YOLO26::_get_detection(std::queue<BBox> &bboxes, int layer_id, float *confidence_cell_buf,
                          float *coordinate_cell_buf, int row, int col, float *confs_tmp)
{
    // process confidence score
    float best_label_score = confidence_cell_buf[0] - 1.f; // arbitrary small number
    int best_label = -1;

    for (size_t label = 0; label < class_count_; label++) {

        if (confidence_cell_buf[label] < confidence_thresh_fastSigmoid_)
            continue;

        if (confidence_cell_buf[label] > best_label_score) {
            best_label_score = confidence_cell_buf[label];
            best_label = label;
        }
    }

    // No score of detection over confidence threshold
    if (best_label == -1)
        return;

    // NOTE: Be aware of the range of fast_signoid: (-1, 1).
    best_label_score = mxutil_prepost_sigmoid_fast_sigmoid(best_label_score);

    // range (-1, 1) -> (0, 1), need to convert, because confidence_thresh is based on range(0, 1)
    best_label_score = (best_label_score + 1.0f) * 0.5f;

    // YOLO26 (YOLOv11): Direct bbox regression with 4 values (left, top, right, bottom) 
    // relative to the anchor point
    float dist_left = coordinate_cell_buf[0];
    float dist_top = coordinate_cell_buf[1];
    float dist_right = coordinate_cell_buf[2];
    float dist_bottom = coordinate_cell_buf[3];

    // Calculate anchor center point in the grid
    float anchor_center_x = (col + 0.5f) * yolo_post_layers_[layer_id].ratio;
    float anchor_center_y = (row + 0.5f) * yolo_post_layers_[layer_id].ratio;

    // Calculate bbox coordinates from anchor point and distances
    float min_x = mxutil_max(anchor_center_x - dist_left * yolo_post_layers_[layer_id].ratio, 0.0f);
    float min_y = mxutil_max(anchor_center_y - dist_top * yolo_post_layers_[layer_id].ratio, 0.0f);
    float max_x = mxutil_min(anchor_center_x + dist_right * yolo_post_layers_[layer_id].ratio, (float)model_input_width_);
    float max_y = mxutil_min(anchor_center_y + dist_bottom * yolo_post_layers_[layer_id].ratio, (float)model_input_height_);

    BBox bbox(best_label, best_label_score, min_x, min_y, max_x, max_y);

    non_maximum_suppression(bboxes, bbox, iou_thresh_);
}

void YOLO26::postprocess(std::vector<float *> output_buffers, YOLO26Result &result)
{
    if (!valid_input_)
    {
        throw std::runtime_error("Make sure to call ComputePadding() before further processing.");
    }

    if (output_buffers.empty())
    {
        throw std::invalid_argument("output_buffers cannot be null.");
    }

    float *confs_tmp = new float[class_count_];

    for (size_t layer_id = 0; layer_id < kNumPostProcessLayers; ++layer_id)
    {
        const auto &layer = yolo_post_layers_[layer_id];
        const int confidence_floats_per_row = (layer.width * class_count_);               // e.g., 80 x 80
        const int coordinate_floats_per_row = (layer.width * layer.coordinate_fmap_size); // e.g., 80 x 4 (YOLO26)

        const int conf_id = layer.confidence_ofmap_flow_id;
        const int coord_id = layer.coordinate_ofmap_flow_id;

        float *confidence_base = output_buffers[conf_id];
        float *coordinate_base = output_buffers[coord_id];

        if (!confidence_base || !coordinate_base)
        {
            throw std::invalid_argument("One or more output buffers are null.");
        }

        for (size_t row = 0; row < layer.height; row++)
        {
            float *confidence_row_buf = confidence_base + row * confidence_floats_per_row;
            float *coordinate_row_buf = coordinate_base + row * coordinate_floats_per_row;

            for (size_t col = 0; col < layer.width; col++)
            {
                float *confidence_cell_buf = confidence_row_buf + col * class_count_;
                float *coordinate_cell_buf = coordinate_row_buf + col * layer.coordinate_fmap_size;
                _get_detection(result.bboxes, layer_id, confidence_cell_buf, coordinate_cell_buf, row, col, confs_tmp);
            }
        }
    }

    delete [] confs_tmp;
}
