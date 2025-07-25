#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/dataset_reader.hpp"
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/ply_loader.hpp"
#include "core/trainer.hpp"
#include "visualizer/detail.hpp"
#include <expected>
#include <print>
#include <thread>

int main(int argc, char* argv[]) {
    //--------------------------------------------------------------------------
    // 1. Parse arguments
    //--------------------------------------------------------------------------
    auto params_result = gs::args::parse_args_and_params(argc, argv);
    if (!params_result) {
        std::println(stderr, "Error: {}", params_result.error());
        return -1;
    }
    auto params = std::move(*params_result);

    //--------------------------------------------------------------------------
    // 2. Check if we're in viewer mode
    //--------------------------------------------------------------------------

    // @TODO: Refactor viewer mode logic into a separate function or class.
    // The current implementation tightly couples viewer mode handling with the main function,
    // making the code less modular and harder to maintain. Extracting this logic will improve
    // code readability and reusability.
    if (params.viewer_mode) {
        // PLY Viewer Mode
        std::println("Loading PLY file: {}", params.ply_path.string());

        auto splat_result = gs::load_ply(params.ply_path);
        if (!splat_result) {
            std::println(stderr, "Error loading PLY: {}", splat_result.error());
            return -1;
        }

        auto splat_data = std::make_unique<SplatData>(std::move(*splat_result));
        std::println("Loaded {} Gaussians", splat_data->size());

        // Create viewer without trainer
        std::string title = "3DGS Viewer - " + params.ply_path.filename().string();
        auto viewer = std::make_unique<gs::GSViewer>(title, 3840, 2160);
        viewer->setTrainer(nullptr); // No trainer in viewer mode
        viewer->setStandaloneModel(std::move(splat_data));
        viewer->setAntiAliasing(params.optimization.antialiasing);

        std::println("Starting viewer...");
        std::println("Anti-aliasing: {}", params.optimization.antialiasing ? "enabled" : "disabled");

        viewer->run();

        std::println("Viewer closed.");
        return 0;
    }

    //--------------------------------------------------------------------------
    // 3. Save training configuration to output directory
    //--------------------------------------------------------------------------
    auto save_result = gs::param::save_training_parameters_to_json(params, params.dataset.output_path);
    if (!save_result) {
        std::println(stderr, "Error saving config: {}", save_result.error());
        return -1;
    }

    //--------------------------------------------------------------------------
    // 4. Create dataset from COLMAP
    //--------------------------------------------------------------------------

    auto dataSetReader = GetValidDataReader(params.dataset);
    auto dataset_result = dataSetReader->create_dataset();

    if (!dataset_result) {
        std::println(stderr, "Error creating dataset: {}", dataset_result.error());
        return -1;
    }
    auto [dataset, scene_center] = std::move(*dataset_result);

    //--------------------------------------------------------------------------
    // 5. Model initialisation
    //--------------------------------------------------------------------------
    auto splat_result = SplatData::init_model_from_pointcloud(params, scene_center, std::move(dataSetReader));
    if (!splat_result) {
        std::println(stderr, "Error initializing model: {}", splat_result.error());
        return -1;
    }
    auto splat_data = std::move(*splat_result);

    //--------------------------------------------------------------------------
    // 6. Create strategy
    //--------------------------------------------------------------------------
    auto strategy = std::make_unique<MCMC>(std::move(splat_data));

    //--------------------------------------------------------------------------
    // 7. Create trainer
    //--------------------------------------------------------------------------
    auto trainer = std::make_unique<gs::Trainer>(
        std::move(dataset),
        std::move(strategy),
        params);

    //--------------------------------------------------------------------------
    // 8. Start training based on visualization mode
    //--------------------------------------------------------------------------
    if (!params.optimization.headless) {
        // GUI Mode: Create viewer and run it in main thread
        auto viewer_result = trainer->create_and_get_viewer();
        if (!viewer_result) {
            std::println(stderr, "Error creating viewer: {}", viewer_result.error());
            return -1;
        }
        auto viewer = *viewer_result;

        // Start training in a separate jthread with stop token support
        std::jthread training_thread([trainer_ptr = trainer.get()](std::stop_token stop_token) {
            auto train_result = trainer_ptr->train(stop_token);
            if (!train_result) {
                std::println(stderr, "Training error: {}", train_result.error());
            }
        });

        // Run GUI in main thread (blocking)
        viewer->run();

        // Request cancellation when GUI closes
        if (trainer->is_running()) {
            std::println("Main: Requesting training stop...");
            training_thread.request_stop();
        }

        // jthread automatically joins when destroyed
        std::println("Main: Waiting for training thread to finish...");
        // training_thread destructor will join automatically
        std::println("Main: Training thread finished.");

    } else {
        // Headless Mode: Run training in main thread
        auto train_result = trainer->train();
        if (!train_result) {
            std::println(stderr, "Training error: {}", train_result.error());
            return -1;
        }
    }

    return 0;
}