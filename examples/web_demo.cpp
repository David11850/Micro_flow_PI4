/**
 * @file web_demo.cpp
 * @brief MicroFlow Web服务 - 手写数字识别HTTP API
 *
 * @启动: ./web_demo <model_path> [port]
 * @访问: http://localhost:8080
 */

#include "microflow/runtime.hpp"
#include "microflow/tensor.hpp"
#include "microflow/image.hpp"
#include "microflow/httplib.h"
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

using namespace microflow;

// 全局推理引擎
InferenceEngine* g_engine = nullptr;

/**
 * @brief 将Tensor转换为JSON字符串
 */
std::string tensor_to_json(const Tensor& output) {
    const float* ptr = output.raw_ptr();
    int num_classes = 10;

    // 找最大值
    float max_val = ptr[0];
    int predicted_digit = 0;
    for (int i = 1; i < num_classes; ++i) {
        if (ptr[i] > max_val) {
            max_val = ptr[i];
            predicted_digit = i;
        }
    }

    std::ostringstream json;
    json << std::fixed << std::setprecision(6);
    json << "{";
    json << "\"digit\":" << predicted_digit << ",";
    json << "\"confidence\":" << max_val << ",";
    json << "\"scores\":[";

    for (int i = 0; i < num_classes; ++i) {
        json << ptr[i];
        if (i < num_classes - 1) json << ",";
    }

    json << "]}";
    return json.str();
}

/**
 * @brief 处理识别请求
 */
void handle_predict(const httplib::Request& req, httplib::Response& res) {
    // 设置CORS
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");

    // 处理OPTIONS预检请求
    if (req.method == "OPTIONS") {
        res.status = 200;
        return;
    }

    try {
        // 解析JSON请求
        // 格式: {"pixels": [0.1, 0.2, ..., 784个值]}
        auto body = req.body;

        // 简单解析（生产环境应使用json库）
        size_t pixels_start = body.find("\"pixels\":");
        if (pixels_start == std::string::npos) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid request format\"}", "application/json");
            return;
        }

        pixels_start = body.find("[", pixels_start);
        size_t pixels_end = body.find("]", pixels_start);

        if (pixels_start == std::string::npos || pixels_end == std::string::npos) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid pixels format\"}", "application/json");
            return;
        }

        // 提取像素数据
        std::string pixels_str = body.substr(pixels_start + 1, pixels_end - pixels_start - 1);
        std::vector<float> pixels;
        std::stringstream ss(pixels_str);
        std::string token;

        while (std::getline(ss, token, ',')) {
            // 去除空白
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                pixels.push_back(std::stof(token));
            }
        }

        // 验证数据长度
        if (pixels.size() != 784) {
            res.status = 400;
            res.set_content("{\"error\":\"Expected 784 pixels, got " +
                           std::to_string(pixels.size()) + "\"}", "application/json");
            return;
        }

        // 创建输入Tensor
        Tensor input({1, 28, 28});
        std::memcpy(input.raw_ptr(), pixels.data(), 784 * sizeof(float));

        // 执行推理
        Tensor output = g_engine->infer(input);

        // 返回JSON结果
        std::string json_result = tensor_to_json(output);
        res.set_content(json_result, "application/json");

    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", "application/json");
    }
}

/**
 * @brief 处理可视化请求 - 返回中间层激活图
 */
void handle_visualize(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");

    if (req.method == "OPTIONS") {
        res.status = 200;
        return;
    }

    try {
        auto body = req.body;
        size_t pixels_start = body.find("\"pixels\":");
        if (pixels_start == std::string::npos) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid request format\"}", "application/json");
            return;
        }

        pixels_start = body.find("[", pixels_start);
        size_t pixels_end = body.find("]", pixels_start);

        if (pixels_start == std::string::npos || pixels_end == std::string::npos) {
            res.status = 400;
            res.set_content("{\"error\":\"Invalid pixels format\"}", "application/json");
            return;
        }

        std::string pixels_str = body.substr(pixels_start + 1, pixels_end - pixels_start - 1);
        std::vector<float> pixels;
        std::stringstream ss(pixels_str);
        std::string token;

        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t\n\r"));
            token.erase(token.find_last_not_of(" \t\n\r") + 1);
            if (!token.empty()) {
                pixels.push_back(std::stof(token));
            }
        }

        if (pixels.size() != 784) {
            res.status = 400;
            res.set_content("{\"error\":\"Expected 784 pixels, got " + std::to_string(pixels.size()) + "\"}", "application/json");
            return;
        }

        // 创建输入张量
        Tensor input({1, 28, 28});
        std::memcpy(input.raw_ptr(), pixels.data(), 784 * sizeof(float));

        // 执行推理
        Tensor output = g_engine->infer(input);

        // 获取中间层输出
        std::vector<Tensor> intermediates = g_engine->get_intermediate_outputs();

        // 构建JSON响应
        std::ostringstream json;
        json << "{";
        json << "\"digit\":" << output.raw_ptr()[0] << ",";  // 简化：取最大值
        json << "\"layers\":[";

        for (size_t i = 0; i < intermediates.size(); ++i) {
            const Tensor& layer = intermediates[i];
            const auto& shapes = layer.shapes();

            json << "{";
            json << "\"index\":" << i << ",";
            json << "\"shape\":[";
            for (size_t j = 0; j < shapes.size(); ++j) {
                json << shapes[j];
                if (j < shapes.size() - 1) json << ",";
            }
            json << "],";

            // 只返回前64个像素作为预览（避免数据太大）
            json << "\"preview\":[";
            uint32_t preview_size = std::min(static_cast<uint32_t>(64), layer.size());
            const float* ptr = layer.raw_ptr();
            for (uint32_t j = 0; j < preview_size; ++j) {
                json << std::fixed << std::setprecision(4) << ptr[j];
                if (j < preview_size - 1) json << ",";
            }
            json << "]";

            json << "}";
            if (i < intermediates.size() - 1) json << ",";
        }

        json << "]}";
        res.set_content(json.str(), "application/json");

    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content("{\"error\":\"" + std::string(e.what()) + "\"}", "application/json");
    }
}

/**
 * @brief 主页面
 */
void handle_index(const httplib::Request& req, httplib::Response& res) {
    res.set_header("Content-Type", "text/html; charset=utf-8");
    res.set_content(R"(
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MicroFlow 手写数字识别</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 500px;
            width: 100%;
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 28px;
        }
        .canvas-wrapper {
            position: relative;
            margin: 0 auto 30px;
            width: 280px;
            height: 280px;
        }
        #canvas {
            border: 3px solid #667eea;
            border-radius: 10px;
            cursor: crosshair;
            background: white;
            touch-action: none;
        }
        .result {
            text-align: center;
            margin: 20px 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
        }
        .result-digit {
            font-size: 72px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .result-confidence {
            font-size: 18px;
            opacity: 0.9;
        }
        .buttons {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        button {
            padding: 15px 30px;
            font-size: 16px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            font-weight: 600;
        }
        #recognize {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        #recognize:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        #clear {
            background: #f0f0f0;
            color: #333;
        }
        #clear:hover {
            background: #e0e0e0;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            margin: 10px 0;
        }
        .loading.show {
            display: block;
        }
        .info {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>✍️ MicroFlow 手写识别</h1>

        <div class="canvas-wrapper">
            <canvas id="canvas" width="280" height="280"></canvas>
        </div>

        <div class="result" id="result" style="display: none;">
            <div class="result-digit" id="digit">?</div>
            <div class="result-confidence" id="confidence">置信度: 0%</div>
        </div>

        <div class="loading" id="loading">
            正在识别...
        </div>

        <div class="buttons">
            <button id="recognize">识别</button>
            <button id="visualize">可视化</button>
            <button id="clear">清空</button>
        </div>

        <p class="info">请在上方区域手写 0-9 的数字</p>

        <!-- 可视化结果 -->
        <div id="visualization" style="display: none; text-align: center; margin-top: 20px;">
            <p style="font-size: 14px; color: #666; margin-bottom: 10px;">AI 看到的中间层激活图：</p>
            <div id="layers" style="display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;"></div>
        </div>

        <!-- 调试预览 -->
        <div style="text-align: center; margin-top: 20px;">
            <p style="font-size: 12px; color: #999;">发送给模型的图像 (28x28):</p>
            <canvas id="preview" width="140" height="140" style="border: 1px solid #ccc; margin: 0 auto;"></canvas>
        </div>
    </div>

    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        let lastX = 0, lastY = 0;

        // 初始化画布
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, 280, 280);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 20;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // 鼠标事件
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);

        // 触摸事件
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouchMove);
        canvas.addEventListener('touchend', stopDrawing);

        function startDrawing(e) {
            isDrawing = true;
            const rect = canvas.getBoundingClientRect();
            lastX = e.clientX - rect.left;
            lastY = e.clientY - rect.top;
        }

        function draw(e) {
            if (!isDrawing) return;
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();

            lastX = x;
            lastY = y;
        }

        function stopDrawing() {
            isDrawing = false;
        }

        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            isDrawing = true;
            lastX = touch.clientX - rect.left;
            lastY = touch.clientY - rect.top;
        }

        function handleTouchMove(e) {
            e.preventDefault();
            if (!isDrawing) return;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;

            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();

            lastX = x;
            lastY = y;
        }

        // 清空画布
        document.getElementById('clear').addEventListener('click', function() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, 280, 280);
            document.getElementById('result').style.display = 'none';
            document.getElementById('visualization').style.display = 'none';
        });

        // 可视化中间层
        document.getElementById('visualize').addEventListener('click', async function() {
            const loading = document.getElementById('loading');
            const visDiv = document.getElementById('visualization');
            const layersDiv = document.getElementById('layers');

            loading.classList.add('show');
            visDiv.style.display = 'none';

            try {
                const imageData = ctx.getImageData(0, 0, 280, 280);
                const pixels = compressTo28x28(imageData);

                const response = await fetch('/visualize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pixels: pixels })
                });

                const data = await response.json();

                // 显示每一层
                layersDiv.innerHTML = '';
                for (let i = 0; i < data.layers.length; i++) {
                    const layer = data.layers[i];
                    const layerDiv = document.createElement('div');
                    layerDiv.style.textAlign = 'center';

                    const label = document.createElement('p');
                    label.textContent = 'Layer ' + i;
                    label.style.fontSize = '12px';
                    label.style.margin = '5px 0';

                    const canvas = document.createElement('canvas');
                    const shape = layer.shape;

                    // 根据层形状确定画布大小
                    if (shape.length === 3) {
                        const h = shape[1];
                        const w = shape[2];
                        const scale = Math.min(100 / h, 100 / w);
                        canvas.width = w * scale;
                        canvas.height = h * scale;

                        const ctx2 = canvas.getContext('2d');
                        const imgData = ctx2.createImageData(w, h);

                        // 填充预览数据
                        for (let j = 0; j < Math.min(layer.preview.length, w * h); j++) {
                            const val = Math.floor(layer.preview[j] * 255);
                            imgData.data[j * 4] = val;
                            imgData.data[j * 4 + 1] = val;
                            imgData.data[j * 4 + 2] = val;
                            imgData.data[j * 4 + 3] = 255;
                        }
                        ctx2.putImageData(imgData, 0, 0);
                    }

                    layerDiv.appendChild(label);
                    layerDiv.appendChild(canvas);
                    layersDiv.appendChild(layerDiv);
                }

                visDiv.style.display = 'block';
            } catch (error) {
                alert('可视化失败: ' + error.message);
            } finally {
                loading.classList.remove('show');
            }
        });

        // 识别
        document.getElementById('recognize').addEventListener('click', async function() {
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');

            loading.classList.add('show');
            resultDiv.style.display = 'none';

            try {
                // 获取图像数据并压缩到28x28
                const imageData = ctx.getImageData(0, 0, 280, 280);
                const pixels = compressTo28x28(imageData);

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pixels: pixels })
                });

                const data = await response.json();

                document.getElementById('digit').textContent = data.digit;
                document.getElementById('confidence').textContent =
                    '置信度: ' + (data.confidence * 100).toFixed(1) + '%';
                resultDiv.style.display = 'block';

            } catch (error) {
                alert('识别失败: ' + error.message);
            } finally {
                loading.classList.remove('show');
            }
        });

        // 压缩图像到28x28
        function compressTo28x28(imageData) {
            const srcData = imageData.data;
            const output = new Array(784);

            // 使用双线性插值缩放到28x28
            const scaleX = 280 / 28;
            const scaleY = 280 / 28;

            for (let y = 0; y < 28; y++) {
                for (let x = 0; x < 28; x++) {
                    // 对应源图像的中心位置
                    const srcX = Math.floor(x * scaleX + scaleX / 2);
                    const srcY = Math.floor(y * scaleY + scaleY / 2);

                    if (srcX < 280 && srcY < 280) {
                        const idx = (srcY * 280 + srcX) * 4;
                        // RGB转灰度，然后反色
                        const gray = (srcData[idx] + srcData[idx + 1] + srcData[idx + 2]) / 3 / 255;
                        output[y * 28 + x] = 1.0 - gray;  // 反色
                    } else {
                        output[y * 28 + x] = 1.0;  // 背景（黑色）
                    }
                }
            }

            // 更新预览画布
            updatePreview(output);

            return output;
        }

        // 更新预览画布
        function updatePreview(pixels) {
            const previewCanvas = document.getElementById('preview');
            const previewCtx = previewCanvas.getContext('2d');
            const imgData = previewCtx.createImageData(28, 28);

            for (let i = 0; i < 784; i++) {
                const val = Math.floor(pixels[i] * 255);
                const idx = i * 4;
                imgData.data[idx] = val;     // R
                imgData.data[idx + 1] = val; // G
                imgData.data[idx + 2] = val; // B
                imgData.data[idx + 3] = 255; // A
            }

            // 创建临时画布绘制28x28，然后缩放到140x140显示
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(imgData, 0, 0);

            // 缩放显示
            previewCtx.imageSmoothingEnabled = false;
            previewCtx.drawImage(tempCanvas, 0, 0, 140, 140);
        }
    </script>
</body>
</html>
    )", "text/html");
}

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║     MicroFlow Web Server                  ║\n";
    std::cout << "║     手写数字识别 HTTP 服务                 ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // 检查参数
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path> [port]\n\n";
        std::cout << "Arguments:\n";
        std::cout << "  model_path  - Path to .mflow model file\n";
        std::cout << "  port        - HTTP server port (default: 8080)\n\n";
        std::cout << "Example:\n";
        std::cout << "  " << argv[0] << " models/mnist_mixed.mflow 8080\n\n";
        return 1;
    }

    std::string model_path = argv[1];
    int port = 8080;
    if (argc > 2) {
        port = std::atoi(argv[2]);
    }

    // 创建并配置推理引擎
    std::cout << "Initializing inference engine...\n";
    InferenceEngine::Config config;
    config.num_threads = 4;

    static InferenceEngine engine(config);
    g_engine = &engine;

    // 加载模型
    std::cout << "Loading model from: " << model_path << "\n";
    if (!engine.load_model(model_path)) {
        std::cerr << "\nError: Failed to load model!\n\n";
        return 1;
    }
    std::cout << "Model loaded successfully!\n\n";

    // 创建HTTP服务器
    httplib::Server svr;

    // 注册路由
    svr.Get("/", handle_index);
    svr.Post("/predict", handle_predict);
    svr.Post("/visualize", handle_visualize);

    // 启动服务器
    std::cout << "╔════════════════════════════════════════════╗\n";
    std::cout << "║  Server starting...                         ║\n";
    std::cout << "╚════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "  URL: http://localhost:" << port << "\n";
    std::cout << "  API: POST /predict\n";
    std::cout << "       {\"pixels\": [0.1, 0.2, ..., 784 values]}\n\n";
    std::cout << "  Press Ctrl+C to stop\n\n";

    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "Failed to start server on port " << port << "\n";
        return 1;
    }

    return 0;
}
