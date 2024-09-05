// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_dot_backward_device_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_dot_backward {
MorehDotBackwardOperation::program_factory_t MorehDotBackwardOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For now we litteraly don't care and return a single factory. Whatever
    return SingleCore{};
}

void MorehDotBackwardOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TT_FATAL("INTENTIONAL: validate_on_program_cache_miss");
}

void MorehDotBackwardOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // TT_FATAL("INTENTIONAL: validate_on_program_cache_hit");
}

void grad_tensor_validate(const Tensor& tensor, const Tensor& grad_tensor) {
    const auto& tensor_shape = tensor.get_shape().value.without_padding();
    const auto& grad_tensor_shape = grad_tensor.get_shape().value.without_padding();
    TT_ASSERT(tensor_shape == grad_tensor_shape);
    TT_ASSERT(grad_tensor.storage_type() == StorageType::DEVICE, "Operands to dot backward need to be on device!");
    TT_ASSERT(grad_tensor.device() == tensor.device(), "Operands to dot backward need to be on the same device!");
    TT_ASSERT(grad_tensor.buffer() != nullptr, "Operands to dot backward need to be allocated in buffers on device!");
}

void MorehDotBackwardOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& other = tensor_args.other;

    TT_ASSERT(tt::operations::primary::is_scalar(output_grad));
    TT_ASSERT(tt::operations::primary::is_1d_tensor(input));
    TT_ASSERT(tt::operations::primary::is_1d_tensor(other));
    TT_ASSERT(tt::operations::primary::is_same_shape(input, other));

    TT_ASSERT(
        input.get_dtype() == DataType::BFLOAT16 || input.get_dtype() == DataType::BFLOAT8_B, "Unsupported data format");
    TT_ASSERT(
        output_grad.storage_type() == StorageType::DEVICE and input.storage_type() == StorageType::DEVICE and
            other.storage_type() == StorageType::DEVICE,
        "Operands to dot backward need to be on device!");
    TT_ASSERT(
        output_grad.device() == input.device() and input.device() == other.device(),
        "Operands to dot backward need to be on the same device!");
    TT_ASSERT(
        output_grad.buffer() != nullptr and input.buffer() != nullptr and other.buffer() != nullptr,
        "Operands to dot backward need to be allocated in buffers on device!");

    const auto& input_grad = tensor_args.input_grad;
    const auto& other_grad = tensor_args.other_grad;
    if (input_grad) {
        grad_tensor_validate(input, input_grad.value());
    }
    if (other_grad) {
        grad_tensor_validate(other, other_grad.value());
    }
}

MorehDotBackwardOperation::shape_return_value_t MorehDotBackwardOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    return {};
}

MorehDotBackwardOperation::tensor_return_value_t MorehDotBackwardOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {

    return {};
}

std::tuple<MorehDotBackwardOperation::operation_attributes_t, MorehDotBackwardOperation::tensor_args_t>
MorehDotBackwardOperation::invoke(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& other,
    std::optional<const Tensor> input_grad,
    std::optional<const Tensor> other_grad,
    const MemoryConfig& mem_config) {
    return {
        operation_attributes_t{mem_config},
        tensor_args_t{output_grad, input, other, input_grad, other_grad}
    };
}

}
