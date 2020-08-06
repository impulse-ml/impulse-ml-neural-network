#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            namespace Optimizer {

                class Abstract {
                protected:
                    T_Size batchSize = 0;
                    T_Size t = 0;
                    double learningRate = 0.1;
                public:
                    void setBatchSize(T_Size);

                    void setT(T_Size);

                    void setLearningRate(double);

                    virtual void optimize(Layer::Abstract *) = 0;
                };
            }
        }
    }
}