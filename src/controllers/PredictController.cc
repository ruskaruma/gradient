#include "controllers/PredictController.h"
#include "engine/InferenceEngine.h"
#include <simdjson.h>
#include <drogon/drogon.h>

namespace gradient {

void PredictController::handlePrediction(const drogon::HttpRequestPtr& req,
                                         std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
    std::string_view body = req->body();
    const std::string& content_type = req->getHeader("Content-Type");
    
    std::pair<int, float> result;
    
    if (content_type.find("application/octet-stream") != std::string::npos) {
        size_t float_count = body.size() / sizeof(float);
        const float* float_ptr = reinterpret_cast<const float*>(body.data());
        std::span<float> input_span(const_cast<float*>(float_ptr), float_count);
        
        result = InferenceEngine::Generic()->predict(input_span);
        
    } else {
        static thread_local simdjson::dom::parser parser;
        simdjson::dom::element doc = parser.parse(body.data(), body.size());
        
        std::vector<float> input_buffer;
        if (doc.is_array()) {
            for (double val : doc) {
                input_buffer.push_back(static_cast<float>(val));
            }
        } else {
             for (double val : doc["input"]) {
                input_buffer.push_back(static_cast<float>(val));
            }
        }
        
        std::span<float> input_span(input_buffer);
        result = InferenceEngine::Generic()->predict(input_span);
    }

    Json::Value json_resp;
    json_resp["class_id"] = result.first;
    json_resp["confidence"] = result.second;
    
    auto resp = drogon::HttpResponse::newHttpJsonResponse(json_resp);
    callback(resp);
}

}
