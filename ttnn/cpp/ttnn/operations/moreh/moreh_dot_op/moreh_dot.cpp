// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot.hpp"
#include "ttnn/operations/moreh/moreh_dot_op/device/moreh_dot_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_dot {

    Tensor MorehDot::invoke(
        const ttnn::Tensor& input_tensor_a,
        const ttnn::Tensor& input_tensor_b,
        const DataType output_dtype,
        const std::optional<MemoryConfig> &output_memory_config) {
            return ttnn::prim::moreh_dot(
                input_tensor_a, input_tensor_b, output_dtype, output_memory_config);
        }
}
