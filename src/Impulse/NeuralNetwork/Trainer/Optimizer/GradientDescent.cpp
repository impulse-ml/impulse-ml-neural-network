#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void GradientDescent::optimize(Layer::Abstract *layer) {
                    Computation::factory().gradientDescent(layer, this->learningRate);
                }
            }
        }
    }
}