#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class MiniBatchGradientDescent : public AbstractTrainer {
            protected:
                T_Size batchSize = 100;
                double beta1 = 0.9;
                double beta2 = 0.999;
                double epsilon = 1e-8;
            public:
                explicit MiniBatchGradientDescent(Network::Abstract &net);

                void setBatchSize(T_Size value);

                void train(Impulse::Dataset::SlicedDataset &dataSet) override;
            };
        }
    }
}
