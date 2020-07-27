#ifndef IMPULSE_VECTORIZED_BACKPROPAGATIONPOOLTOCONV_H
#define IMPULSE_VECTORIZED_BACKPROPAGATIONPOOLTOCONV_H

#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                class BackPropagationToConv : public Abstract {
                public:
                    BackPropagationToConv(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    Math::T_Matrix propagate(const Math::T_Matrix &input,
                                             T_Size numberOfExamples,
                                             double regularization,
                                             const Math::T_Matrix &sigma) override;
                };
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_BACKPROPAGATIONPOOLTOCONV_H
