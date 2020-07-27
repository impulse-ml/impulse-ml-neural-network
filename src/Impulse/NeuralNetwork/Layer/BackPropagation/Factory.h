#ifndef IMPULSE_VECTORIZED_BACKPROPAGATION_H
#define IMPULSE_VECTORIZED_BACKPROPAGATION_H

#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                class Factory {
                public:
                    static BackPropagationPointer create(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);
                };
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_BACKPROPAGATION_H
