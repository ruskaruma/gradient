#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <span>
#include <memory>
#include <mutex>

namespace gradient {

class InferenceEngine {
public:
    static InferenceEngine* Generic();

    InferenceEngine(const InferenceEngine&) = delete;
    InferenceEngine& operator=(const InferenceEngine&) = delete;

    void loadModel(const std::string& path);

    std::pair<int, float> predict(const std::span<float>& input_data);

private:
    InferenceEngine();
    ~InferenceEngine() = default;

    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    
    std::vector<std::vector<char>> input_node_names_buffers_;
    std::vector<std::vector<char>> output_node_names_buffers_;
    std::vector<const char*> input_node_names_ptrs_;
    std::vector<const char*> output_node_names_ptrs_;
};

}
