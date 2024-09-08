// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    int i{0};
    const auto num_tiles = get_arg_val<uint32_t>(i++);

    std::uint8_t input_id{0};
    const auto cb_x = input_id++;                  // input
    const auto cb_clip_coef_clamped = input_id++;  // clip_coef_clamped

    std::uint8_t output_id{16};
    const auto cb_y = output_id++;  // output

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(cb_x, cb_clip_coef_clamped);

    cb_wait_front(cb_clip_coef_clamped, onetile);  // comes from the reader

    // Compute cb_y
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        ACQ();
        cb_wait_front(cb_x, onetile);  // comes from the reader
        cb_reserve_back(cb_y, onetile);

        mul_tiles_bcast_scalar_init_short();
        mul_tiles_bcast_scalar(cb_x, cb_clip_coef_clamped, 0, 0, dst0);

        pack_tile(dst0, cb_y);

        cb_pop_front(cb_x, onetile);
        cb_push_back(cb_y, onetile);
        REL();
    }

    cb_pop_front(cb_clip_coef_clamped, onetile);
}  // void MAIN
}  // namespace NAMESPACE
