#ifndef IMPULSE_VECTORIZED_BACKPROPAGATION1DTO3D_H
#define IMPULSE_VECTORIZED_BACKPROPAGATION1DTO3D_H

#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                class BackPropagationToMaxPool : public Abstract {
                public:
                    BackPropagationToMaxPool(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    Math::T_Matrix propagate(const Math::T_Matrix &input,
                                             T_Size numberOfExamples,
                                             double regularization,
                                             const Math::T_Matrix &sigma) override;
                };
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_BACKPROPAGATION1DTO1D_H
