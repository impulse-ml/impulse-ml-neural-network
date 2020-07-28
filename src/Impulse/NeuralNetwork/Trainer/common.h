#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult {
                double cost;
                double accuracy;
                Eigen::VectorXd gradient;

                double &getCost() {
                    return this->cost;
                }

                double &getAccuracy() {
                    return this->accuracy;
                }

                Eigen::VectorXd &getGradient() {
                    return this->gradient;
                }
            };

            typedef std::function<CostGradientResult(Eigen::VectorXd)> StepFunction;
        }
    }
}
