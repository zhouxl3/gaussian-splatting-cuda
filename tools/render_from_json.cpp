#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/ply_loader.hpp"
#include "core/rasterizer.hpp"
#include <args.hxx>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

struct CameraInfo {
    int uid;
    torch::Tensor R;
    torch::Tensor T;
    float fov_x;
    float fov_y;
    torch::Tensor principal;
    int width;
    int height;
    std::string name;
};

static CameraInfo parse_camera_json(const nlohmann::json& cam_json, int idx)
{
    CameraInfo info{};
    info.uid = idx;
    info.name = cam_json.value("img_id", std::to_string(idx));

    auto intr = cam_json["intrinsics"];
    auto ext = cam_json["extrinsics"]["c2w_matrix"];
    info.width = cam_json.value("width", 0);
    info.height = cam_json.value("height", 0);

    torch::Tensor c2w = torch::empty({4, 4}, torch::kFloat32);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) { c2w[i][j] = static_cast<float>(ext[i][j]); }
    }

    // c2w[:3, 1:3] *= -1
    c2w.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(1, 3)}) *=
        -1.f;

    torch::Tensor w2c = torch::inverse(c2w);

    // R is stored transposed due to glm in the CUDA code
    info.R = w2c.index({torch::indexing::Slice(0, 3),
                        torch::indexing::Slice(0, 3)}).t();
    info.T = w2c.index({torch::indexing::Slice(0, 3), 3});

    float fx = intr[0][0];
    float fy = intr[1][1];
    float cx = intr[0][2];
    float cy = intr[1][2];

    info.fov_x = 2.f * std::atan(info.width / (2.f * fx));
    info.fov_y = 2.f * std::atan(info.height / (2.f * fy));

    info.principal =
        torch::tensor({1.f - cx / info.width, 1.f - cy / info.height});

    return info;
}

int main(int argc, char** argv) {
    args::ArgumentParser parser(
        "Render Gaussian Splat from JSON cameras",
        "Renders a PLY Gaussian model using cameras provided in a JSON array.");

    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> ply_arg(
        parser, "ply", "Path to the Gaussian PLY model", {'p', "ply"});
    args::ValueFlag<std::string> json_arg(
        parser, "json", "Camera JSON file", {'j', "json"});
    args::ValueFlag<std::string> out_arg(
        parser, "output", "Output directory", {'o', "output"});

    try {
        parser.ParseCLI(argc, argv);
    } catch (const args::Help&) {
        std::cout << parser.Help();
        return 0;
    } catch (const args::ParseError& e) {
        std::cerr << e.what() << '\n'
                  << parser.Help();
        return 1;
    }

    if (!ply_arg || !json_arg || !out_arg) {
        std::cerr << parser.Help();
        return 1;
    }

    std::filesystem::path ply_path = args::get(ply_arg);
    std::filesystem::path json_path = args::get(json_arg);
    std::filesystem::path out_dir = args::get(out_arg);

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

        CameraInfo info = parse_camera_json(cam_json, cam_idx);

        float fx = cam_json["intrinsics"][0][0];
        float fy = cam_json["intrinsics"][1][1];
        float cx = cam_json["intrinsics"][0][2];
        float cy = cam_json["intrinsics"][1][2];
        Camera cam(info.R.t(),
                   info.T,
                   fx,
                   fy,
                   cx,
                   cy,
                   torch::empty({0}, torch::kFloat32),
                   torch::empty({0}, torch::kFloat32),
                   gsplat::CameraModelType::PINHOLE,
                   info.name,
                   "",
                   info.width,
                   info.height,
                   cam_idx);

        auto output = gs::rasterize(cam, model, bg_color);
        std::filesystem::path out_path = out_dir / (info.name + ".png");
        save_image(out_path, output.image);
        std::cout << "Saved " << out_path << "\n";
        ++cam_idx;
    }

    return 0;
}
