#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                Abstract::Abstract(Layer::LayerPointer layer, Layer::LayerPointer previousLayer) {
                    this->layer = layer;
                    this->previousLayer = previousLayer;
                }
            }
        }
    }
}