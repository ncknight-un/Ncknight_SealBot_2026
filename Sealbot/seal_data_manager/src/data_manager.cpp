// Features:
// 1. Visualise camera frame on RViz.
// 2. Subscribes to record camera frame.
// 3. Publishes motor commands.
// 4. Publishes joystick commands.


#include <exception>
#include <vector>
#include <string>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/image_encodings.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/string.hpp>
#include <thread>
#include <mutex>

using namespace std::chrono_literals;

class DataManager : public rclcpp::Node
{
public:
    DataManager() : Node("data_manager")
    {
        RCLCPP_INFO(this->get_logger(), "Data Manager Node Initialized");

        // Publishers
        camera_pub = this->create_publisher<sensor_msgs::msg::Image>("camera/image", 10);
        motor_pub = this->create_publisher<geometry_msgs::msg::Twist>("motor/commands", 10);
        joystick_pub = this->create_publisher<geometry_msgs::msg::Twist>("joystick/commands", 10);
        
        // Subscribers
        // camera_sub = this->create_subscription<sensor_msgs::msg::Image>("camera/image", 10, std::bind(&DataManager::image_callback, this, std::placeholders::_1));

        // Joystick.
        // Motor command publishers.

        // Thread for reading camera stream
        cam_thread = std::thread(&DataManager::capture_camera_stream, this);

        // Timer
        mTimer = this->create_wall_timer(100ms, std::bind(&DataManager::timer_callback, this)); 
    }

    ~DataManager()
    {
        stop_camera = true;
        if (cam_thread.joinable())
        {
            cam_thread.join();
        } 
    }

    // void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    // {
    //     RCLCPP_INFO(this->get_logger(), "Received camera frame");
    //     // Process the image and publish it
    //     camera_pub->publish(*msg);
    // }

    void capture_camera_stream()
    {
        // Use GStreamer pipeline for UDP H264 stream from BlueOS
        std::string gst_pipeline = 
            "udpsrc port=5600 caps=\"application/x-rtp\" ! "
            "rtph264depay ! avdec_h264 ! videoconvert ! appsink";

        cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);

        if (!cap.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open camera stream");
            return;
        }

        RCLCPP_INFO(this->get_logger(), "Camera stream opened");

        cv::Mat frame;
        sensor_msgs::msg::Image::SharedPtr ros_image;

        while (rclcpp::ok() && !stop_camera)
        {
            cap >> frame;
            if (frame.empty()) continue;

            std_msgs::msg::Header header;
            header.stamp = get_clock()->now();
            header.frame_id = "camera_frame";  // or something like "camera_link"

            ros_image = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
            ros_image->header.stamp = this->now();
            camera_pub->publish(*ros_image);

            std::this_thread::sleep_for(33ms); // ~30 fps
        }
    }

    void timer_callback()
    {
        RCLCPP_INFO(this->get_logger(), "Timer callback executed");

        // // Publish motor commands
        // auto motor_msg = geometry_msgs::msg::Twist();
        // motor_msg.linear.x = 0.2;
        // motor_msg.angular.z = 0.1;
        // motor_pub_->publish(motor_msg);

        // // Publish joystick commands
        // auto joystick_msg = geometry_msgs::msg::Twist();
        // joystick_msg.linear.y = 0.1;
        // joystick_msg.angular.z = -0.1;
        // joystick_pub_->publish(joystick_msg);

    }

private:
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_pub;
    // image_transport::Publisher camera_pub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr motor_pub;
    // rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr camera_sub;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr joystick_pub;
    std::thread cam_thread;
    std::atomic<bool> stop_camera{false};
    rclcpp::TimerBase::SharedPtr mTimer;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DataManager>());
    rclcpp::shutdown();
    return 0;
}