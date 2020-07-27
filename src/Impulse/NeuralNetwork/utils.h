#ifndef IMPULSE_VECTORIZED_UTILS_H
#define IMPULSE_VECTORIZED_UTILS_H

#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Utils {

            Math::T_Matrix im2col(const Math::T_Matrix &input, int channels,
                                  int height, int width,
                                  int kernel_h, int kernel_w,
                                  int pad_h, int pad_w,
                                  int stride_h, int stride_w);

            Math::T_Matrix maxpool(const Math::T_Matrix &input, int channels,
                                   int height, int width,
                                   int kernel_h, int kernel_w,
                                   int stride_h, int stride_w);
        }
    }
}

#endif //IMPULSE_VECTORIZED_UTILS_H
