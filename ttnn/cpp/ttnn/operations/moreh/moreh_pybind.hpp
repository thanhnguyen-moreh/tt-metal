// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "ttnn/operations/moreh/moreh_dot_op_backward/moreh_dot_backward_pybind.hpp"


namespace py = pybind11;

namespace ttnn::operations::moreh {
    void py_module(py::module& module);
}
