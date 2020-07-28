#pragma once

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
