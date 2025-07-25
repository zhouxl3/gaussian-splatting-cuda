#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/ply_loader.hpp"
#include "core/rasterizer.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <ply_file> <camera_json> <output_dir>\n";
        return 1;
    }
    std::filesystem::path ply_path = argv[1];
    std::filesystem::path json_path = argv[2];
    std::filesystem::path out_dir = argv[3];

    if (!std::filesystem::exists(ply_path)) {
        std::cerr << "PLY file not found: " << ply_path << "\n";
        return 1;
    }
    if (!std::filesystem::exists(json_path)) {
        std::cerr << "Camera JSON file not found: " << json_path << "\n";
        return 1;
    }
    std::filesystem::create_directories(out_dir);

    auto splat_result = gs::load_ply(ply_path);
    if (!splat_result) {
        std::cerr << "Failed to load PLY: " << splat_result.error() << "\n";
        return 1;
    }
    SplatData model = std::move(*splat_result);

    std::ifstream json_stream(json_path);
    nlohmann::json json;
    json_stream >> json;

    std::vector<nlohmann::json> cameras;
    if (json.is_array()) {
        cameras.assign(json.begin(), json.end());
    } else if (json.is_object()) {
        cameras.push_back(json);
    } else {
        std::cerr << "Invalid JSON format\n";
        return 1;
    }

    torch::Tensor bg_color = torch::zeros({3});
    int cam_idx = 0;
    for (const auto& cam_json : cameras) {
        if (!cam_json.contains("intrinsics") || !cam_json.contains("extrinsics")) {
            std::cerr << "Camera entry missing intrinsics or extrinsics\n";
            continue;
        }
        auto intr = cam_json["intrinsics"];
        auto ext = cam_json["extrinsics"]["c2w_matrix"];
        int width = cam_json.value("width", 0);
        int height = cam_json.value("height", 0);
        std::string img_id = cam_json.value("img_id", std::to_string(cam_idx));

        torch::Tensor c2w = torch::empty({4, 4}, torch::kFloat32);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                c2w[i][j] = static_cast<float>(ext[i][j]);
            }
        }
        torch::Tensor w2c = torch::inverse(c2w);
        torch::Tensor R = w2c.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});
        torch::Tensor T = w2c.index({torch::indexing::Slice(0, 3), 3});

        float fx = intr[0][0];
        float fy = intr[1][1];
        float cx = intr[0][2];
        float cy = intr[1][2];

        Camera cam(R,
                   T,
                   fx,
                   fy,
                   cx,
                   cy,
                   torch::empty({0}, torch::kFloat32),
                   torch::empty({0}, torch::kFloat32),
                   gsplat::CameraModelType::PINHOLE,
                   img_id,
                   "",
                   width,
                   height,
                   cam_idx);

        auto output = gs::rasterize(cam, model, bg_color);
        std::filesystem::path out_path = out_dir / (img_id + ".png");
        save_image(out_path, output.image);
        std::cout << "Saved " << out_path << "\n";
        ++cam_idx;
    }

    return 0;
}
