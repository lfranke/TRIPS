/**
 * Copyright (c) 2024 Darius Rückert and Linus Franke
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "spherical_harmonics.h"

#include <saiga/vision/torch/TorchHelper.h>

SphericalHarmonicsEncoding::SphericalHarmonicsEncoding(int degree) : degree(degree)
{
    using tcnn::cpp::json;

    json config;

    // These settings are from here and are used for nerf-like problems
    // https://github.com/NVlabs/instant-ngp/blob/master/configs/nerf/base.json
    config["otype"]  = "SphericalHarmonics";
    config["degree"] = degree;

    auto net_module = tcnn::cpp::create_encoding(3, config, tcnn::cpp::EPrecision::Fp32);

    module = TcnnTorchModule(TorchTcnnWrapperModule(net_module));

    std::cout << "params: " << module->params.numel() << std::endl;
}
torch::Tensor SphericalHarmonicsEncoding::forward(torch::Tensor direction)
{
    auto lin                    = direction.reshape({-1, 3});
    int size_original_flattened = lin.size(0);

    uint32_t granularity = tcnn::cpp::batch_size_granularity();

    // Saiga::PrintTensorInfo(direction);
    //  scale to the unit cube [0, 1]
    lin = lin * 0.5 + 0.5;
    lin = torch::nn::functional::pad(
        lin, torch::nn::functional::PadFuncOptions({0, 0, 0, granularity - (size_original_flattened % granularity)})
                 .mode(torch::kConstant));
    torch::Tensor x;
    x = module->forward(lin.contiguous());


    CHECK(x.defined());
    CHECK_EQ(x.dim(), lin.dim());
    CHECK_EQ(x.size(0), lin.size(0));

    x = x.slice(0, 0, size_original_flattened);

    auto out_shape   = direction.sizes().vec();
    out_shape.back() = x.size(-1);
    x                = x.reshape(out_shape);

    return x;
}
