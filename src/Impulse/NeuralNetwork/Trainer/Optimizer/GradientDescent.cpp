#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void GradientDescent::optimize(Layer::Abstract *layer) {
                    layer->getComputation()->gradientDescent(this->learningRate);
                }
            }
        }
    }
}