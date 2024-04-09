// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/tt_metal.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;
    tt_metal::InterleavedBufferConfig dram_config{
                .device= device,
                .size = single_tile_size,
                .page_size = single_tile_size,
                .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src0_vec(1, 14);
    std::vector<uint32_t> src1_vec(1, 7);

    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

    /* Use L1 circular buffers to set input buffers */
    constexpr uint32_t src0_cb_index = CB::c_in0;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CB::c_in1;
    CircularBufferConfig cb_src1_config = CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src1_cb_index, single_tile_size);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    /* Specify data movement kernel for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {
            src0_dram_buffer->address(),
            src1_dram_buffer->address(),
            dst_dram_buffer->address(),
        }
    );

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
    printf("Result = %d : Expected = 21\n", result_vec[0]);

    CloseDevice(device);
}
