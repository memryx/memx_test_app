#include <iostream>
#include <thread>
#include <signal.h>
#include <opencv2/opencv.hpp>    /* imshow */
#include <opencv2/imgproc.hpp>   /* cvtcolor */
#include <opencv2/imgcodecs.hpp> /* imwrite */
#include <chrono>
#include <algorithm>
#include <atomic>
#include <deque>
#include <filesystem>
#include <mutex>
#include <vector>
#include "memx/accl/MxAccl.h"
#include "yolo26.h"

namespace fs = std::filesystem;

std::atomic_bool runflag;  // Atomic flag to control run state
bool is_show_global = true;  // Global flag to indicate if any stream should show display - ALWAYS TRUE
int num_streams_global = 2;  // Total number of configured streams - ALWAYS 2
std::mutex display_mutex;  // Protect shared OpenCV display state

// Quadrant window configuration
int screen_width = 0;
int screen_height = 0;
int quadrant_width = 0;
int quadrant_height = 0;
cv::Mat logo_image;
cv::Mat modules_image;

// Yolo26 application specific parameters - HARDCODED FOR SILLY GOOSE MODE
fs::path model_path = "YOLO26_nano_640_640_3_onnx.dfp";  // Hardcoded DFP path
std::string video_str = "cam:0,cam:2";  // Hardcoded dual camera setup

#define FPS_LOG_INTERVAL 30  // print out FPS every X frames
#define FRAME_QUEUE_MAX_LENGTH     5

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal) {
    runflag.store(false);  // Stop the program
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
                    // Add FPS/stream text to the display (truncated to integer)
                    std::string fps_text = "Stream " + std::to_string(stream_idx) +
                                           " FPS: " + std::to_string(static_cast<int>(fps_number));
                    cv::putText(display_image, fps_text, cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

                    std::lock_guard<std::mutex> dlock(display_mutex);

                    // Display each stream in its own quadrant window
                    std::string window_name = (stream_idx == 0) ? "Stream 0 - Camera 0" : "Stream 1 - Camera 2";
                    
                    // Resize to fit quadrant and display
                    cv::Mat resized_display;
                    cv::resize(display_image, resized_display, cv::Size(quadrant_width, quadrant_height));
                    cv::imshow(window_name, resized_display);

                    // Handle keyboard input
                    int key = cv::waitKey(1);
                    if (key == 27 || key == 'q' || key == 'Q') {  // ESC or Q key to quit
                        runflag.store(false);
                    }
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
    std::cout << "YOLO26 Silly Goose Quadrant Mode Starting..." << std::endl;
    
    try {
        signal(SIGINT, signal_handler);  // Set up signal handler
        
        // HARDCODED CONFIGURATION - SILLY GOOSE MODE!
        std::vector<std::string> video_src_list = {"cam:0", "cam:2"};
        int num_streams = 2;
        num_streams_global = num_streams;
        
        // Hardcoded screen resolution for embedded platform
        // To change: edit these two lines
        screen_width = 1920;
        screen_height = 1080;
        
        // Calculate quadrant dimensions
        quadrant_width = screen_width / 2;   // Should be 960
        quadrant_height = screen_height / 2; // Should be 540
        
        std::cout << "Screen Resolution: " << screen_width << "x" << screen_height << std::endl;
        std::cout << "Quadrant Size: " << quadrant_width << "x" << quadrant_height << std::endl;
        std::cout << "  Width: " << quadrant_width << " (should be WIDER)" << std::endl;
        std::cout << "  Height: " << quadrant_height << " (should be SHORTER)" << std::endl;
        
        // Get current working directory for debugging
        fs::path cwd = fs::current_path();
        std::cout << "Current working directory: " << cwd << std::endl;
        
        // Try to find images in current directory or executable directory
        fs::path exe_path = fs::path(argv[0]).parent_path();
        std::cout << "Executable path: " << exe_path << std::endl;
        
        // Try multiple locations for the images
        std::vector<fs::path> search_paths = {
            cwd / "combined_logos.png",
            exe_path / "combined_logos.png",
            cwd / ".." / "combined_logos.png"
        };
        
        // Load logo image
        logo_image = cv::Mat();
        for (const auto& path : search_paths) {
            std::cout << "Trying to load logo from: " << path << std::endl;
            if (fs::exists(path)) {
                logo_image = cv::imread(path.string());
                if (!logo_image.empty()) {
                    std::cout << "Successfully loaded logo from: " << path << std::endl;
                    break;
                }
            }
        }
        
        if (logo_image.empty()) {
            std::cerr << "Warning: Could not load combined_logos.png from any location, using placeholder" << std::endl;
            logo_image = cv::Mat(quadrant_height, quadrant_width, CV_8UC3, cv::Scalar(50, 50, 50));
            cv::putText(logo_image, "combined_logos.png", cv::Point(50, quadrant_height/2),
                       cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 255), 3);
        } else {
            cv::resize(logo_image, logo_image, cv::Size(quadrant_width, quadrant_height));
        }
        
        // Try multiple locations for modules image
        std::vector<fs::path> search_paths_modules = {
            cwd / "combined_modules.png",
            exe_path / "combined_modules.png",
            cwd / ".." / "combined_modules.png"
        };
        
        // Load modules image
        modules_image = cv::Mat();
        for (const auto& path : search_paths_modules) {
            std::cout << "Trying to load modules from: " << path << std::endl;
            if (fs::exists(path)) {
                modules_image = cv::imread(path.string());
                if (!modules_image.empty()) {
                    std::cout << "Successfully loaded modules from: " << path << std::endl;
                    break;
                }
            }
        }
        
        if (modules_image.empty()) {
            std::cerr << "Warning: Could not load combined_modules.png from any location, using placeholder" << std::endl;
            modules_image = cv::Mat(quadrant_height, quadrant_width, CV_8UC3, cv::Scalar(70, 70, 70));
            cv::putText(modules_image, "combined_modules.png", cv::Point(50, quadrant_height/2),
                       cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(255, 255, 255), 3);
        } else {
            cv::resize(modules_image, modules_image, cv::Size(quadrant_width, quadrant_height));
        }

        std::cout << "\n=== SILLY GOOSE QUADRANT CONFIGURATION ===" << std::endl;
        std::cout << "Model path: " << model_path << " (HARDCODED)" << std::endl;
        std::cout << "Number of streams: " << num_streams << " (HARDCODED)" << std::endl;
        std::cout << "Display enabled: YES (ALWAYS)" << std::endl;
        for (int i = 0; i < num_streams; ++i) {
            std::cout << "  Stream " << i << ": " << video_src_list[i] << std::endl;
        }
        std::cout << "==========================================\n" << std::endl;

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

    // Create OpenCV windows positioned at each quadrant
    std::cout << "Creating quadrant windows..." << std::endl;
    
    // Create windows with AUTOSIZE (no manual resizing, minimal decorations)
    cv::namedWindow("Stream 0 - Camera 0", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Stream 1 - Camera 2", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Logos", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Modules", cv::WINDOW_AUTOSIZE);
    
    // Try to remove window decorations (may not work on all window managers)
    cv::setWindowProperty("Stream 0 - Camera 0", cv::WND_PROP_TOPMOST, 1);
    cv::setWindowProperty("Stream 1 - Camera 2", cv::WND_PROP_TOPMOST, 1);
    cv::setWindowProperty("Logos", cv::WND_PROP_TOPMOST, 1);
    cv::setWindowProperty("Modules", cv::WND_PROP_TOPMOST, 1);
    
    // Note: With AUTOSIZE, windows will size to match image content automatically
    // No need to call cv::resizeWindow()
    
    // Wait a moment for window manager to process
    cv::waitKey(50);
    
    // Position windows at each quadrant
    // Top-left: Stream 0
    cv::moveWindow("Stream 0 - Camera 0", 0, 0);
    
    // Top-right: Stream 1  
    cv::moveWindow("Stream 1 - Camera 2", quadrant_width, 0);
    
    // Bottom-left: Logos
    cv::moveWindow("Logos", 0, quadrant_height);
    
    // Bottom-right: Modules
    cv::moveWindow("Modules", quadrant_width, quadrant_height);
    
    // Final wait to let window manager settle
    cv::waitKey(50);
    
    std::cout << "Quadrant windows created and positioned!" << std::endl;
    std::cout << "  Stream 0 at (0, 0) - size " << quadrant_width << "x" << quadrant_height << std::endl;
    std::cout << "  Stream 1 at (" << quadrant_width << ", 0) - size " << quadrant_width << "x" << quadrant_height << std::endl;
    std::cout << "  Logos at (0, " << quadrant_height << ") - size " << quadrant_width << "x" << quadrant_height << std::endl;
    std::cout << "  Modules at (" << quadrant_width << ", " << quadrant_height << ") - size " << quadrant_width << "x" << quadrant_height << std::endl;
    
    // Display static images once - they'll stay visible until program exits
    cv::imshow("Logos", logo_image);
    cv::imshow("Modules", modules_image);
    cv::waitKey(1);  // Process window events once

    std::cout << "Starting accelerator..." << std::endl;
    
    // Run the accelerator and wait
    accl.start();
    accl.wait();
    accl.stop();

    std::cout << "Accelerator stopped." << std::endl;
    
    // Cleanup OpenCV windows
    try {
        cv::destroyAllWindows();
    } catch (...) {
        // Ignore cleanup errors
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
    
    std::cout << "\n🦆 Silly Goose Quadrant Mode finished successfully! 🦆" << std::endl;
    return 0;
    
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal unknown error occurred" << std::endl;
        return 1;
    }
}