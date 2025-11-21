#include <drogon/drogon.h>
#include "engine/InferenceEngine.h"
#include <iostream>

int main() {
    std::string ip = "0.0.0.0";
    uint16_t port = 8080;

    std::cout << "Starting Gradient on " << ip << ":" << port << std::endl;

    gradient::InferenceEngine::Generic()->loadModel("models/resnet50.onnx");

    drogon::app()
        .addListener(ip, port)
        .setThreadNum(0)
        .run();

    return 0;
}
