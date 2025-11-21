#include "engine/InferenceEngine.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace gradient {

InferenceEngine* InferenceEngine::Generic()
{
    static InferenceEngine instance;
    return &instance;
}
InferenceEngine::InferenceEngine() 
    : env_(ORT_LOGGING_LEVEL_ERROR, "Gradient") {
}

void InferenceEngine::loadModel(const std::string& path) {
    Ort::SessionOptions session_options;
    
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, path.c_str(), session_options);

    size_t num_input_nodes = session_.GetInputCount();
    size_t num_output_nodes = session_.GetOutputCount();

    input_names_.clear();
    output_names_.clear();
    input_node_names_ptrs_.clear();
    output_node_names_ptrs_.clear();
    
    for (size_t i = 0; i < num_input_nodes; i++)
    {
        Ort::AllocatedStringPtr name_ptr = session_.GetInputNameAllocated(i, allocator_);
        input_names_.emplace_back(name_ptr.get());
        input_node_names_ptrs_.push_back(input_names_.back().c_str());
    }

    for (size_t i = 0; i < num_output_nodes; i++) {
        Ort::AllocatedStringPtr name_ptr = session_.GetOutputNameAllocated(i, allocator_);
        output_names_.emplace_back(name_ptr.get());
        output_node_names_ptrs_.push_back(output_names_.back().c_str());
    }
}

std::pair<int, float> InferenceEngine::predict(const std::span<float>& input_data) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_node_names_ptrs_.data(),
        &input_tensor,
        1,
        output_node_names_ptrs_.data(),
        output_node_names_ptrs_.size()
    );

    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    int best_idx = 0;
    float best_val = -1.0f;

    for (size_t i = 0; i < output_size; ++i) {
        if (floatarr[i] > best_val) {
            best_val = floatarr[i];
            best_idx = static_cast<int>(i);
        }
    }

    return {best_idx, best_val};
}

}
