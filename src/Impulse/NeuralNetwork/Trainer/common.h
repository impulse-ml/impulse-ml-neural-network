#ifndef IMPULSE_NEURALNETWORK_TRAINER_COMMON_H
#define IMPULSE_NEURALNETWORK_TRAINER_COMMON_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult {
                double cost;
                double accuracy;
                Math::T_Vector gradient;

                double &getCost() {
                    return this->cost;
                }

                double &getAccuracy() {
                    return this->accuracy;
                }

                Math::T_Vector &getGradient() {
                    return this->gradient;
                }
            };

            typedef std::function<CostGradientResult(Math::T_Vector)> StepFunction;
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_TRAINER_COMMON_H
