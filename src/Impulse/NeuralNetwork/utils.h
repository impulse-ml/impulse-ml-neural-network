#pragma once

#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Utils {

            Eigen::MatrixXd im2col(const Eigen::MatrixXd &input, int channels,
                                   int height, int width,
                                   int kernel_h, int kernel_w,
                                   int pad_h, int pad_w,
                                   int stride_h, int stride_w);

            Eigen::MatrixXd maxpool(const Eigen::MatrixXd &input, int channels,
                                    int height, int width,
                                    int kernel_h, int kernel_w,
                                    int stride_h, int stride_w);
        }
    }
}
