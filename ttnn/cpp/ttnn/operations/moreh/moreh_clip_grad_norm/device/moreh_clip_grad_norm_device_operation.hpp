// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"


namespace ttnn::operations::moreh::moreh_clip_grad_norm {

std::tuple<uint32_t, float, bool> get_p_decimal_p_is_negative(float ord);

struct MorehClipGradNormStep1 {
    struct operation_attributes_t {
        float norm_type;
        uint32_t tile_offset_of_tmp_pow_sum;
    };

    struct tensor_args_t {
        const std::vector<Tensor> &input_tensors;
        const std::vector<std::optional<const Tensor>> &optional_input_tensors;
        const Tensor &tmp_pow_sum;
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

    // To do: Implement step 1 impl
    // Note: Read implementations on cpp files
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        float norm_type,
        uint32_t tile_offset_of_tmp_pow_sum,
        const Tensor &tmp_pow_sum);
};

void moreh_clip_grad_norm_step1(const std::vector<Tensor> &input_tensors, float norm_type, const Tensor &tmp_pow_sum);

struct MorehClipGradNormStep2 {
    struct operation_attributes_t {
        float norm_type;
    };
    struct tensor_args_t {
        const Tensor &tmp_pow_sum;
        const Tensor &total_norm;
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

    // To do: Implement step 1 impl
    // Note: Read implementations on cpp files
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor &tmp_pow_sum,
        const Tensor &total_norm,
        float norm_type);
};

void moreh_clip_grad_norm_step2(const Tensor &tmp_pow_sum, float norm_type, const Tensor &total_norm);

struct MorehClipGradNormStep3 {
    struct operation_attributes_t {
    };
    struct tensor_args_t {
        const std::vector<Tensor> &input_tensors;
        const std::vector<std::optional<const Tensor>> &optional_input_tensors;
        const Tensor &clip_coef_clamped;
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

    // To do: Implement step 1 impl
    // Note: Read implementations on cpp files
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const Tensor &clip_coef_clamped);
};

void moreh_clip_grad_norm_step3(const std::vector<Tensor> &inputs, const Tensor &clip_coef_clamped);

Tensor moreh_clip_grad_norm_impl(
    const std::vector<Tensor> &inputs,
    float max_norm,
    float norm_type,
    bool error_if_nonfinite,
    const Tensor &tmp_pow_sum,
    const Tensor &total_norm);

[[maybe_unused]] Tensor moreh_clip_grad_norm(
    const std::vector<Tensor> &inputs,
    float max_norm,
    float norm_type,
    bool error_if_nonfinite,
    const std::optional<std::reference_wrapper<const Tensor>> total_norm,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


}
