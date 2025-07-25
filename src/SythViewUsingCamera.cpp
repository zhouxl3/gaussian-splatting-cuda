#include "core/argument_parser.hpp"
#include "core/dataset.hpp"
#include "core/mcmc.hpp"
#include "core/parameters.hpp"
#include "core/trainer.hpp"
#include "visualizer/detail.hpp"
#include <expected>
#include <print>
#include <thread>



int main(int argc, char* argv[]) {

        auto output = gs::rasterize(
        cam,
        trainer_->get_strategy().get_model(),
        background,
        config_->scaling_modifier,
        false,
        anti_aliasing_,
        RenderMode::RGB);
}