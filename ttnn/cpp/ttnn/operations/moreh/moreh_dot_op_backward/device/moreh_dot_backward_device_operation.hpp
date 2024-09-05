// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"


namespace ttnn::operations::moreh::moreh_dot_backward {

struct MorehDotBackwardOperation {
    struct operation_attributes_t {
        const MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor &output_grad;
        const Tensor &input;
        const Tensor &other;
        const std::optional<const Tensor> &input_grad;
        const std::optional<const Tensor> &other_grad;
    };

    using shape_return_value_t = std::vector<ttnn::Shape>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct SingleCore {
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SingleCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);


    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor &output_grad,
        const Tensor &input,
        const Tensor &other,
        std::optional<const Tensor> input_grad,
        std::optional<const Tensor> other_grad,
        const MemoryConfig &mem_config);
    };
}

namespace ttnn::prim {
constexpr auto moreh_dot_backward = ttnn::register_operation<"ttnn::prim::moreh_dot_backward", ttnn::operations::moreh::moreh_dot_backward::MorehDotBackwardOperation>();
}  // namespace ttnn
