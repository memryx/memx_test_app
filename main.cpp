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
#include <condition_variable>
#include <vector>
#include "memx/accl/MxAccl.h"
#include "yolo26.h"

namespace fs = std::filesystem;

// ── Canvas dimensions ────────────────────────────────────────────────────────
static constexpr int DISP_W = 1280;
static constexpr int DISP_H = 720;
static constexpr int HALF_W = DISP_W / 2;
static constexpr int HALF_H = DISP_H / 2;

std::atomic_bool runflag;  // Atomic flag to control run state
bool is_show_global = true;  // Global flag to indicate if any stream should show display - ALWAYS TRUE
int num_streams_global = 2;  // Total number of configured streams - ALWAYS 2

// ── Display globals (double-buffering with dedicated display thread) ─────────
cv::Mat           g_canvas[2];
std::atomic<int>  g_back_idx{0};
std::mutex        g_roi_mutex[2];        // per-stream, guards ROI write into back canvas
std::mutex        g_disp_mutex;
std::condition_variable g_disp_cv;
std::atomic<bool> g_frame_ready{false};
std::atomic<bool> is_fullscreen{false};

// Image paths for static tiles
std::string logo_image_path;
std::string modules_image_path;

// Yolo26 application specific parameters
fs::path model_path = "YOLO26_nano_640_640_3_onnx.dfp";  // Hardcoded DFP path
std::string video_str = "cam:0,cam:2";  // Hardcoded dual camera setup

#define FPS_LOG_INTERVAL 30  // print out FPS every X frames
#define FRAME_QUEUE_MAX_LENGTH     5

// Signal handler to gracefully stop the program on SIGINT (Ctrl+C)
void signal_handler(int p_signal) {
    runflag.store(false);  // Stop the program
    g_disp_cv.notify_all();
}

// ── Helper: Paint static tiles ───────────────────────────────────────────────
void paint_static_tiles(cv::Mat& canvas,
                        const std::string& logo_path,
                        const std::string& modules_path)
{
    auto paint = [](cv::Mat& roi, const std::string& path, const std::string& label) {
        if (!path.empty()) {
            cv::Mat img = cv::imread(path);
            if (!img.empty()) {
                double scale = std::min((double)roi.cols / img.cols,
                                        (double)roi.rows / img.rows);
                int nw = (int)(img.cols * scale);
                int nh = (int)(img.rows * scale);
                cv::Mat sub = roi(cv::Rect((roi.cols - nw) / 2, (roi.rows - nh) / 2, nw, nh));
                cv::resize(img, sub, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);
                return;
            }
        }
        // Fallback: dark background with label
        roi.setTo(cv::Scalar(30, 30, 30));
        cv::putText(roi, label, cv::Point(roi.cols / 2 - 60, roi.rows / 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
    };

    cv::Mat tl = canvas(cv::Rect(0,      0,      HALF_W, HALF_H));
    cv::Mat br = canvas(cv::Rect(HALF_W, HALF_H, HALF_W, HALF_H));
    paint(tl, logo_path,    "Logos");
    paint(br, modules_path, "Modules");

    cv::line(canvas, cv::Point(HALF_W, 0),      cv::Point(HALF_W, DISP_H), cv::Scalar(200, 200, 200), 2);
    cv::line(canvas, cv::Point(0,      HALF_H), cv::Point(DISP_W, HALF_H), cv::Scalar(200, 200, 200), 2);
}

// ── Dedicated display thread ─────────────────────────────────────────────────
// All HighGUI calls live here so GTK sees them on one consistent thread.
void display_thread_func() {
    cv::namedWindow("YOLO26 Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("YOLO26 Detection", DISP_W, DISP_H);

    // Pre-paint static tiles now that the window (and GTK context) is owned by this thread.
    for (int i = 0; i < 2; i++)
        paint_static_tiles(g_canvas[i], logo_image_path, modules_image_path);

    using clock = std::chrono::steady_clock;
    const std::chrono::milliseconds frame_interval{33}; // ~30 fps display rate
    auto next_display = clock::now();

    while (runflag.load()) {
        {
            std::unique_lock<std::mutex> lk(g_disp_mutex);
            g_disp_cv.wait_for(lk, frame_interval, [] {
                return g_frame_ready.load() || !runflag.load();
            });
            g_frame_ready.store(false, std::memory_order_relaxed);
        }

        if (!runflag.load())
            break;

        // Rate-limit to ~30fps: if we displayed too recently, just pump GTK events and loop.
        // This drains burst notifications from two streams without swapping on every one.
        auto now = clock::now();
        if (now < next_display) {
            cv::waitKey(1);
            continue;
        }
        next_display = now + frame_interval;

        // Swap back→front, drain in-flight ROI writes, then present.
        int front = g_back_idx.load();
        g_back_idx.store(1 - front, std::memory_order_release);

        {
            std::lock_guard<std::mutex> lk0(g_roi_mutex[0]);
            std::lock_guard<std::mutex> lk1(g_roi_mutex[1]);
            cv::imshow("YOLO26 Detection", g_canvas[front]);
        }

        int key = cv::waitKey(1);
        if (key == 'f' || key == 'F') {
            bool fs = !is_fullscreen.load();
            is_fullscreen.store(fs);
            cv::setWindowProperty("YOLO26 Detection", cv::WND_PROP_FULLSCREEN,
                fs ? cv::WINDOW_FULLSCREEN : cv::WINDOW_NORMAL);
        } else if (key == 'q' || key == 'Q' || key == 27) {
            runflag.store(false);
        }
    }
    cv::destroyAllWindows();
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

            // Display the updated image using dedicated display thread
            if (is_show) {
                try {
                    // Add FPS/stream text to the display (truncated to integer)
                    std::string fps_text = "Stream " + std::to_string(stream_idx) +
                                           " FPS: " + std::to_string(static_cast<int>(fps_number));
                    cv::putText(display_image, fps_text, cv::Point(10, 30),
                               cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

                    // Write directly into back canvas ROI — no separate windows
                    int back = g_back_idx.load(std::memory_order_relaxed);
                    cv::Mat roi = g_canvas[back](
                        cv::Rect(stream_idx == 0 ? 0 : HALF_W, stream_idx == 0 ? HALF_H : 0, HALF_W, HALF_H));
                    {
                        std::lock_guard<std::mutex> lk(g_roi_mutex[stream_idx]);
                        cv::resize(display_image, roi, cv::Size(HALF_W, HALF_H), 0, 0, cv::INTER_LINEAR);
                    }
                    g_frame_ready.store(true, std::memory_order_release);
                    g_disp_cv.notify_one();
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
    std::cout << "YOLO26 Detection Application Starting..." << std::endl;
    
    try {
        signal(SIGINT, signal_handler);  // Set up signal handler
        
        // Hardcoded configuration
        std::vector<std::string> video_src_list = {"cam:0", "cam:2"};
        int num_streams = 2;
        num_streams_global = num_streams;
        
        std::cout << "Display Canvas: " << DISP_W << "x" << DISP_H << std::endl;
        std::cout << "Quadrant Size: " << HALF_W << "x" << HALF_H << std::endl;
        
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
        
        // Find logo image path
        for (const auto& path : search_paths) {
            std::cout << "Trying to load logo from: " << path << std::endl;
            if (fs::exists(path)) {
                cv::Mat test = cv::imread(path.string());
                if (!test.empty()) {
                    logo_image_path = path.string();
                    std::cout << "Successfully found logo at: " << path << std::endl;
                    break;
                }
            }
        }
        
        if (logo_image_path.empty()) {
            std::cerr << "Warning: Could not load combined_logos.png from any location" << std::endl;
        }
        
        // Try multiple locations for modules image
        std::vector<fs::path> search_paths_modules = {
            cwd / "combined_modules.png",
            exe_path / "combined_modules.png",
            cwd / ".." / "combined_modules.png"
        };
        
        // Find modules image path
        for (const auto& path : search_paths_modules) {
            std::cout << "Trying to load modules from: " << path << std::endl;
            if (fs::exists(path)) {
                cv::Mat test = cv::imread(path.string());
                if (!test.empty()) {
                    modules_image_path = path.string();
                    std::cout << "Successfully found modules at: " << path << std::endl;
                    break;
                }
            }
        }
        
        if (modules_image_path.empty()) {
            std::cerr << "Warning: Could not load combined_modules.png from any location" << std::endl;
        }

        std::cout << "\n=== CONFIGURATION ===" << std::endl;
        std::cout << "Model path: " << model_path << std::endl;
        std::cout << "Number of streams: " << num_streams << std::endl;
        std::cout << "Display enabled: YES" << std::endl;
        for (int i = 0; i < num_streams; ++i) {
            std::cout << "  Stream " << i << ": " << video_src_list[i] << std::endl;
        }
        std::cout << "=====================" << std::endl;

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

    // Pre-allocate canvases (tiles are painted inside display_thread_func after window creation)
    for (int i = 0; i < 2; i++)
        g_canvas[i] = cv::Mat(DISP_H, DISP_W, CV_8UC3, cv::Scalar(0, 0, 0));

    // Start display thread — owns all HighGUI calls to satisfy GTK's single-thread requirement
    std::thread disp_thread(display_thread_func);
    
    // Give display thread a moment to create window
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

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
            runflag.store(false);
            g_disp_cv.notify_all();
            disp_thread.join();
            return 1;
        }
    }

    std::cout << "Starting accelerator..." << std::endl;
    
    // Run the accelerator and wait
    accl.start();
    accl.wait();
    accl.stop();

    std::cout << "Accelerator stopped." << std::endl;
    
    // Stop display thread
    runflag.store(false);
    g_disp_cv.notify_all();
    disp_thread.join();

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
    
    std::cout << "\nYOLO26 Detection Application finished successfully." << std::endl;
    return 0;
    
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal unknown error occurred" << std::endl;
        return 1;
    }
}