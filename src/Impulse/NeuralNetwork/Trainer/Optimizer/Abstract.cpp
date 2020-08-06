#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                void Abstract::setBatchSize(T_Size size) {
                    this->batchSize = size;
                }

                void Abstract::setT(T_Size t) {
                    this->t = t;
                }

                void Abstract::setLearningRate(double v) {
                    this->learningRate = v;
                }
            }
        }
    }
}