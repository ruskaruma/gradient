#pragma once

#include <drogon/HttpController.h>

namespace gradient {

class PredictController : public drogon::HttpController<PredictController> {
public:
    METHOD_LIST_BEGIN
        METHOD_ADD(PredictController::handlePrediction, "/predict", drogon::Post);
    METHOD_LIST_END

    void handlePrediction(const drogon::HttpRequestPtr& req,
                          std::function<void(const drogon::HttpResponsePtr&)>&& callback);
};

}
