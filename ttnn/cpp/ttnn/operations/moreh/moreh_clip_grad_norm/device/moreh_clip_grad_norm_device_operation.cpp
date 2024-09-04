// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_clip_grad_norm_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"


namespace ttnn::operations::moreh::moreh_clip_grad_norm {

inline uint32_t get_num_device_cores(Device *device) {
    const auto num_cores_x = static_cast<uint32_t>(device->compute_with_storage_grid_size().x);
    const auto num_cores_y = static_cast<uint32_t>(device->compute_with_storage_grid_size().y);
    return num_cores_x * num_cores_y;
}

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord) {
    auto p = std::floor(ord);
    auto decimal = ord - p;
    const bool p_is_negative = p < 0.0f;
    if (p_is_negative) {
        p = -p;
    }
    return std::make_tuple(static_cast<uint32_t>(p), decimal, p_is_negative);
}

MorehClipGradNormStep1::program_factory_t MorehClipGradNormStep1::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCore{};
}

void MorehClipGradNormStep1::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
}

void MorehClipGradNormStep1::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
}

void MorehClipGradNormStep1::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const std::vector<Tensor> &input_tensors = tensor_args.input_tensors;
    const std::vector<std::optional<const Tensor>> &optional_input_tensors = tensor_args.optional_input_tensors;
    for (const auto &input : input_tensors) {
        tt::operations::primary::check_tensor(input, "moreh_clip_grad_norm_step1", "input");
    }

    const auto &tmp_pow_sum = optional_input_tensors.at(0).value();
    tt::operations::primary::check_tensor(tmp_pow_sum, "moreh_clip_grad_norm_step1", "tmp_pow_sum");
}

MorehClipGradNormStep1::shape_return_value_t MorehClipGradNormStep1::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
}

MorehClipGradNormStep1::tensor_return_value_t MorehClipGradNormStep1::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return {};
}


std::tuple<MorehClipGradNormStep1::operation_attributes_t, MorehClipGradNormStep1::tensor_args_t>
MorehClipGradNormStep1::invoke(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<const Tensor>> &optional_input_tensors,
    float norm_type,
    uint32_t tile_offset_of_tmp_pow_sum,
    const Tensor &tmp_pow_sum) {
    return {
        operation_attributes_t{norm_type, tile_offset_of_tmp_pow_sum},
        tensor_args_t{input_tensors, optional_input_tensors, tmp_pow_sum}
    };
}

void moreh_clip_grad_norm_step1(const std::vector<Tensor> &inputs, float norm_type, const Tensor &tmp_pow_sum) {
    auto device = inputs.at(0).device();
    const auto max_num_inputs = get_num_device_cores(device);
    const auto total_num_inputs = static_cast<uint32_t>(inputs.size());

    const auto num_iter = (total_num_inputs + max_num_inputs - 1) / max_num_inputs;

    uint32_t tile_offset{0};
    auto num_inputs = total_num_inputs;
    for (uint32_t i = 0; i < num_iter; ++i) {
        const auto num_inputs_at_this_iter = std::min(num_inputs, max_num_inputs);

        std::vector<Tensor> dummy_output_tensors = {Tensor(operation::get_workers_for_op_output({tmp_pow_sum}))};

        operation::launch_op(
            [norm_type, tile_offset](
                const std::vector<Tensor> &input_tensors,
                const std::vector<std::optional<const Tensor>> &optional_input_tensors,
                const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
                return operation::run(
                    MorehClipGradNormStep1{operation_attributes_t.norm_type = norm_type, .tile_offset_of_tmp_pow_sum = tile_offset},
                    input_tensors,
                    optional_input_tensors,
                    optional_output_tensors);
            },
            std::vector<Tensor>(inputs.begin() + tile_offset, inputs.begin() + tile_offset + num_inputs_at_this_iter),
            dummy_output_tensors,
            {tmp_pow_sum});

        if (i < (num_iter - 1)) {
            tile_offset += num_inputs_at_this_iter;
            num_inputs -= num_inputs_at_this_iter;
        }
    }
}

}  // namespace ttnn::operations::examples
