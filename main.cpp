#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include "memx/accl/MxAccl.h"
#include "yolo26.h"

namespace fs = std::filesystem;

std::atomic_bool runflag;  // Atomic flag to control run state
bool is_show_global = false;  // Global flag to indicate if any stream should show display

// Yolo26 application specific parameters
fs::path model_path = "../../assets/models/YOLO26_nano_640_640_3_onnx.dfp";  // Default model path
std::string video_str = "vid:../../assets/video/sample.mp4";

#define FPS_LOG_INTERVAL 30  // print out FPS every X frames
#define FRAME_QUEUE_MAX_LENGTH     5

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal) {
    runflag.store(false);  // Stop the program
}

// Function to display usage information
void print_usage(const std::string& program_name) {
    std::cout << "Usage: " << program_name
              << " [-d <dfp_path>] [--show] [--video_paths \"cam:0,vid:video_path\"]\n"
              << "Options:\n"
              << "  -d, --dfp_path        (Optional) Path to the DFP. Default: "<< model_path<<"\n"
              << "  --video_paths         (Optional) Video paths in the format \"cam:0,vid:video_path,vid:video2_path\". Default: cam:0\n"
              << "  --show                (Optional) Display the inference result. Default: false\n";
}

// Function to configure camera settings (resolution and FPS)
bool configure_camera(cv::VideoCapture& vcap) {
    bool settings_success = true;
    try {
        // Attempt to set 640x480 resolution and 30 FPS
        if (!vcap.set(cv::CAP_PROP_FRAME_HEIGHT, 480) || 
            !vcap.set(cv::CAP_PROP_FRAME_WIDTH, 640) || 
            !vcap.set(cv::CAP_PROP_FPS, 30)) {
            std::cout << "Setting vcap Failed\n";
            cv::Mat simpleframe;
            if (!vcap.read(simpleframe)) {
                settings_success = false;
            }
        }
    } catch (...) {
        std::cout << "Exception occurred while setting properties\n";
        settings_success = false;
    }
    return settings_success;
}

// Function to open the camera and apply settings, if not possible, reopen with default settings
bool open_camera(cv::VideoCapture& vcap, int device, int api) {
    vcap.open(device, api);  // Open the camera
    if (!vcap.isOpened()) {
        std::cerr << "Failed to open vcap\n";
        return false;
    }

    if (!configure_camera(vcap)) {  // Try applying custom settings
        vcap.release();  // Release and reopen with default settings
        vcap.open(device, api);
        if (vcap.isOpened()) {
            std::cout << "Reopened vcap with original resolution\n";
        } else {
            std::cerr << "Failed to reopen vcap\n";
            return false;
        }
    }
    return true;
}

class YoloApp {
    private:
        // Application Variables
        bool is_show;  // Per-stream display flag
        std::deque<cv::Mat> frames_queue;  // Queue for frames
        std::mutex frame_queue_mutex;  // Mutex to control access to the queue
        
        // FPS related
        int frame_count = 0;
        int total_frame_count = 0;
        float fps_number = .0;  // FPS counter
        std::vector<float> history_fps;
        std::chrono::milliseconds start_ms;

        cv::VideoCapture vcap;  // Video capture object
        bool src_is_cam = false;
        
        MX::Types::MxModelInfo model_info;  // Model info structure
        std::vector<float*> mxa_output;  // Buffer for the output of the accelerator
        std::vector<float*> mxa_input;  // Buffer for the input of the accelerator

        YOLO26* yolo26;  // YOLO26 model
        
        // Input callback function to fetch frames and preprocess them
        bool incallback_getframe(std::vector<const MX::Types::FeatureMap*> dst, int stream_idx) {
            if (runflag.load()) {
                cv::Mat inframe;

                while(true){
                    bool got_frame = vcap.read(inframe);  // Capture frame

                    if (!got_frame) {  // If no frame, stop the stream
                        std::cout << "No frame \n\n\n";
                        vcap.release();
                        return false;
                    }

                    if(src_is_cam && (frames_queue.size() >= FRAME_QUEUE_MAX_LENGTH)){
                        // drop the frame and try again if we've hit the limit
                        continue;
                    }
                    else{
                        // Store original BGR frame in queue for display
                        {
                            std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                            frames_queue.push_back(inframe);
                        }
                    }

                    // Preprocess frame (BGR->RGB conversion happens inside preprocess)
                    cv::Mat preProcframe = yolo26->preprocess(inframe);
                    dst[0]->set_data((float*)preProcframe.data);
                    return true;
                }
            }
            else {
                vcap.release();
                return false;
            }
        }

        // Output callback function to process MXA output and display results
        bool outcallback_getmxaoutput(std::vector<const MX::Types::FeatureMap*> src, int stream_idx) {
            for (int i = 0; i < model_info.num_out_featuremaps; ++i)
            {
                src[i]->get_data(mxa_output[i]);
            }

            cv::Mat display_image;
            {
                std::lock_guard<std::mutex> ilock(frame_queue_mutex);
                display_image = frames_queue.front();
                frames_queue.pop_front();
            }
            
            // Set confidence threshold and process detection results
            YOLO26Result result;
            yolo26->postprocess(mxa_output, result);
            yolo26->draw_result(result, display_image);

            // Display the updated image using OpenCV
            if (is_show) {
                try {
                    // Add FPS text to the display
                    std::string fps_text = "FPS: " + std::to_string(fps_number);
                    cv::putText(display_image, fps_text, cv::Point(10, 30), 
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
                    
                    // Create window name with stream index
                    std::string window_name = "YOLO26 Detection - Stream " + std::to_string(stream_idx);
                    cv::imshow(window_name, display_image);
                    cv::waitKey(1);  // Allow OpenCV to process window events
                } catch (const cv::Exception& e) {
                    std::cerr << "OpenCV display error (continuing): " << e.what() << std::endl;
                    is_show = false;  // Disable further display attempts
                }
            }

            // Calculate FPS
            frame_count++;
            total_frame_count++;
            if (frame_count == 1) {
                start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            }
            else if (frame_count % FPS_LOG_INTERVAL == 0) {
                std::chrono::milliseconds duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()) - start_ms;
                fps_number = (float)FPS_LOG_INTERVAL * 1000 / (float)(duration.count());
                frame_count = 0;
                std::cout << "Frame_count: " << total_frame_count << " Stream " << stream_idx << " FPS: " << fps_number << '\n';
                history_fps.push_back(fps_number);
            }
            return true;
        }

    public:
        float get_avg_fps() const
        {
            float sum = 0;
            for (const auto& fps : history_fps)
            {
                sum += fps;
            }
            return sum / history_fps.size();
        }

        // Constructor to initialize YOLO26 object
        YoloApp(MX::Runtime::MxAccl* accl_, std::string video_src, int stream_idx) {
            is_show = is_show_global;  // Initialize per-stream flag from global
            // Open the camera or video source
            if(video_src.substr(0,3) == "cam") {
                src_is_cam = true;
                int device = std::stoi(video_src.substr(4));
                #ifdef __linux__
                    if (!open_camera(vcap, device, cv::CAP_V4L2)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }
                #elif defined(_WIN32)
                    if (!open_camera(vcap, device, cv::CAP_ANY)) {
                        throw(std::runtime_error("Failed to open: "+video_src));
                    }
                #endif
            } else if (video_src.substr(0,3) == "vid") {
                src_is_cam = false;
                std::cout << "Video source given = " << video_src.substr(4) << "\n\n";
                vcap.open(video_src.substr(4), cv::CAP_ANY);
            } else {
                throw(std::runtime_error("Given video src: " + video_src + " is invalid"));
            }

            if (!vcap.isOpened()) {
                throw(std::runtime_error("Failed to open videocapture for " + video_src));
            }

            // Get input image dimensions
            int input_image_width = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_WIDTH));
            int input_image_height = static_cast<int>(vcap.get(cv::CAP_PROP_FRAME_HEIGHT));
            
            // compute padding
            yolo26 = new YOLO26();
            yolo26->compute_padding(input_image_width, input_image_height);

            // Get model info and allocate input buffer
            model_info = accl_->get_model_info(0 /* model_id */);
            mxa_input.resize(model_info.num_in_featuremaps);
            for (int i = 0; i < model_info.num_in_featuremaps; ++i) {
                mxa_input[i] = new float[model_info.in_featuremap_sizes[i]];
            }
            
            // Get model info and allocate output buffer
            mxa_output.resize(model_info.num_out_featuremaps);
            for (int i = 0; i < model_info.num_out_featuremaps; ++i) {
                mxa_output[i] = new float[model_info.out_featuremap_sizes[i]];
            }

            // Bind input/output callback functions
            auto in_cb = std::bind(&YoloApp::incallback_getframe, this, std::placeholders::_1, std::placeholders::_2);
            auto out_cb = std::bind(&YoloApp::outcallback_getmxaoutput, this, std::placeholders::_1, std::placeholders::_2);

            // Connect streams to the accelerator
            accl_->connect_stream(in_cb, out_cb, stream_idx /**Unique Stream Idx */, 0 /**Model Idx */);

            // Start the input/output streams
            runflag.store(true);
        }

        ~YoloApp() {
            delete yolo26;
            for (int i = 0; i < static_cast<int>(mxa_output.size()); ++i) {
                delete[] mxa_output[i];
            }
            mxa_output.clear();  // Clean up memory
        }
};

int main(int argc, char* argv[]) {
    std::cout << "YOLO26 Application Starting..." << std::endl;
    std::cout << "Arguments: " << argc << std::endl;
    
    try {
        signal(SIGINT, signal_handler);  // Set up signal handler
        std::vector<std::string> video_src_list;

    // Iterate through the arguments
    for (int i = 1; i < argc; i++) {

        std::string arg = argv[i];
        std::cout << "Processing argument: " << arg << std::endl;

        // Handle -d or --dfp_path
        if (arg == "-d" || arg == "--dfp_path") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                model_path = argv[++i];
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        // Handle --video_paths
        else if (arg == "--video_paths") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {  // Ensure there's a next argument and it is not another option
                video_str = argv[++i];
                 size_t pos = 0;
                std::string token;
                std::string delimiter = ",";
                while ((pos = video_str.find(delimiter)) != std::string::npos) {
                    token = video_str.substr(0, pos);
                    video_src_list.push_back(token);
                    video_str.erase(0, pos + delimiter.length());
                }
                video_src_list.push_back(video_str);
            } else {
                std::cerr << "Error: Missing value for " << arg << " option.\n";
                print_usage(argv[0]);
                return 1;
            }
        }
        else if (arg == "--show") {
            is_show_global = true;
        }
        // Handle unknown options
        else {
            std::cerr << "Error: Unknown option " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // if video_paths arg isn't passed - use default video string.
    int num_streams = video_src_list.size();
    if(num_streams == 0){
        video_src_list.push_back(video_str);
        num_streams = 1;
    }  

    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Number of streams: " << num_streams << std::endl;
    std::cout << "Display enabled: " << (is_show_global ? "yes" : "no") << std::endl;
    for (int i = 0; i < num_streams; ++i) {
        std::cout << "  Stream " << i << ": " << video_src_list[i] << std::endl;
    }
    std::cout << "=====================\n" << std::endl;

    std::cout << "Initializing MemryX Accelerator..." << std::endl;
    
    // Init Accl object
    std::vector<int> device_ids = {0};
    std::array<bool, 2> use_model_shape = {false, false};
    bool local_mode = true;
    MX::Runtime::MxAccl accl{fs::path(model_path), device_ids, use_model_shape, local_mode};
    accl.set_num_workers(1, 3, 0);

    std::cout << "Accelerator initialized. Creating " << num_streams << " stream(s)..." << std::endl;
    if (is_show_global) {
        std::cout << "Display enabled (--show flag set)" << std::endl;
    }

    // Creating YoloApp objects for each video stream
    std::vector<YoloApp*> apps;
    for (int i = 0; i < num_streams; ++i) {
        std::cout << "Creating stream " << i << " for video source: " << video_src_list[i] << std::endl;
        try {
            YoloApp* app = new YoloApp(&accl, video_src_list[i], i);
            apps.push_back(app);
            std::cout << "Stream " << i << " created successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating stream " << i << ": " << e.what() << std::endl;
            // Clean up any created apps
            for (auto* a : apps) delete a;
            return 1;
        }
    }

    std::cout << "Starting accelerator..." << std::endl;
    
    // Run the accelerator and wait
    accl.start();
    accl.wait();
    accl.stop();

    std::cout << "Accelerator stopped." << std::endl;
    
    // Cleanup OpenCV windows if display was enabled
    if (is_show_global) {
        try {
            cv::destroyAllWindows();
        } catch (...) {
            // Ignore cleanup errors
        }
    }

    // Print out final avg FPS
    float final_fps = 0.f;
    for (int i = 0; i < num_streams; ++i) {
        float fps = apps[i]->get_avg_fps();
        final_fps += fps;
    }
    std::cout << "Final Avg FPS: " << final_fps << std::endl;

    // Cleanup
    for (int i = 0; i < num_streams; ++i) {
        delete apps[i];
    }
    
    std::cout << "Application finished successfully!" << std::endl;
    return 0;
    
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal unknown error occurred" << std::endl;
        return 1;
    }
}
