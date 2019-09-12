#ifndef _REID_H
#define _REID_H

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include "chrono"
#include <samples/classification_results.h>

class ReID {

public:
    ReID() = default;

    ~ReID();

public:
    void init(const std::string &modelPath, int gpuID, int decrypt_model, bool support_nv12);

    void classify(const std::vector<cv::Mat> &images, std::vector<std::vector<float>> &classify_results);

    void release();

    void set_model_type(int model_type);

private:

    void classify_single(const cv::Mat& images, std::vector<float> &classify_results);


private:
    std::string input_name_;
    std::string output_name_;
    InferenceEngine::CNNNetReader network_reader_;
    InferenceEngine::InferencePlugin plugin_;
    InferenceEngine::CNNNetwork network_;
    InferenceEngine::InputInfo::Ptr input_info_;
    InferenceEngine::DataPtr output_info_;
    std::map<std::string, std::string> networkConfig_;
    InferenceEngine::ExecutableNetwork executable_network_;
    InferenceEngine::InferRequest infer_request_;
};

#endif //_REID_H
