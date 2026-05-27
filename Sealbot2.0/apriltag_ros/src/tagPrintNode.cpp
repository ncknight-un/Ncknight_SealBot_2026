#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/synchronizer.hpp>
#include <opencv2/opencv.hpp>

class DetectionVisualizer : public rclcpp::Node
{
public:
    DetectionVisualizer(const rclcpp::NodeOptions& options)
    : Node("detection_visualizer", options)
    {
        image_sub_.subscribe(this, "/flir_camera/image_raw", rclcpp::QoS(10));
        detections_sub_.subscribe(this, "/apriltag/detections", rclcpp::QoS(10));

        sync_ = std::make_shared<Sync>(SyncPolicy(10), image_sub_, detections_sub_);
        sync_->registerCallback(std::bind(&DetectionVisualizer::callback, this,
                                          std::placeholders::_1, std::placeholders::_2));

        pub_ = image_transport::create_publisher(this, "detections_printout");

        RCLCPP_INFO(get_logger(), "Detection visualizer running. Publishing on /detections_printout");
    }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        apriltag_msgs::msg::AprilTagDetectionArray>;
    using Sync = message_filters::Synchronizer<SyncPolicy>;

    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<apriltag_msgs::msg::AprilTagDetectionArray> detections_sub_;
    std::shared_ptr<Sync> sync_;
    image_transport::Publisher pub_;

    void callback(
        const sensor_msgs::msg::Image::ConstSharedPtr& img_msg,
        const apriltag_msgs::msg::AprilTagDetectionArray::ConstSharedPtr& det_msg)
    {
        RCLCPP_INFO(get_logger(), "callback fired, encoding: %s, detections: %zu",
                    img_msg->encoding.c_str(), det_msg->detections.size());

        // Convert to OpenCV, handling Bayer encoding from FLIR camera
        cv_bridge::CvImagePtr cv_img;
        try {
            cv_img = cv_bridge::toCvCopy(img_msg, "");  // keep original encoding
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // Convert Bayer to BGR if needed
        const std::string& enc = cv_img->encoding;
        if (enc == "bayer_rggb8" || enc == "BayerRG8" || enc == "bayer_rggb16") {
            cv::Mat bgr;
            cv::cvtColor(cv_img->image, bgr, cv::COLOR_BayerRG2BGR);
            cv_img->image = bgr;
            cv_img->encoding = "bgr8";
        } else if (enc == "bayer_bggr8") {
            cv::Mat bgr;
            cv::cvtColor(cv_img->image, bgr, cv::COLOR_BayerBG2BGR);
            cv_img->image = bgr;
            cv_img->encoding = "bgr8";
        } else if (enc == "bayer_gbrg8") {
            cv::Mat bgr;
            cv::cvtColor(cv_img->image, bgr, cv::COLOR_BayerGB2BGR);
            cv_img->image = bgr;
            cv_img->encoding = "bgr8";
        } else if (enc == "bayer_grbg8") {
            cv::Mat bgr;
            cv::cvtColor(cv_img->image, bgr, cv::COLOR_BayerGR2BGR);
            cv_img->image = bgr;
            cv_img->encoding = "bgr8";
        }
        // If already bgr8/rgb8/mono8 etc, cv_bridge handles it fine

        cv::Mat& frame = cv_img->image;

        for (const auto& det : det_msg->detections) {
            std::vector<cv::Point> corners(4);
            for (int i = 0; i < 4; i++) {
                corners[i] = cv::Point(
                    static_cast<int>(det.corners[i].x),
                    static_cast<int>(det.corners[i].y)
                );
            }

            // Draw border
            for (int i = 0; i < 4; i++) {
                cv::line(frame, corners[i], corners[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
            }

            // Draw corner dots
            for (const auto& c : corners) {
                cv::circle(frame, c, 4, cv::Scalar(0, 0, 255), -1);
            }

            // Find top-center of tag
            int min_y = corners[0].y;
            int sum_x = 0;
            for (const auto& c : corners) {
                if (c.y < min_y) min_y = c.y;
                sum_x += c.x;
            }
            int center_x = sum_x / 4;
            cv::Point text_pos(center_x - 20, min_y - 10);

            // Average side lengths in pixels
            auto dist = [](cv::Point a, cv::Point b) {
                return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
            };
            double num_pix = (dist(corners[0], corners[1]) +
                              dist(corners[1], corners[2]) +
                              dist(corners[2], corners[3]) +
                              dist(corners[3], corners[0])) / 4.0;

            std::string label = "ID:" + std::to_string(det.id) +
                                " --- Pix: " + std::to_string(static_cast<int>(num_pix));

            cv::putText(frame, label, text_pos,
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(255, 0, 255), 2);
        }

        pub_.publish(cv_img->toImageMsg());
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectionVisualizer>(rclcpp::NodeOptions()));
    rclcpp::shutdown();
    return 0;
}