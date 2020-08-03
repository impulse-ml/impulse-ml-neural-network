#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            //T_String OPTIMIZER_ADAM = "adam";
            //T_String OPTIMIZER_RMS_PROP = "rmsprop";

            class MiniBatchGradientDescent : public AbstractTrainer {
            protected:
                T_Size batchSize = 100;
                T_String optimizer = "";
            public:
                explicit MiniBatchGradientDescent(Network::Abstract &net);

                void setBatchSize(T_Size value);

                void train(Impulse::Dataset::SlicedDataset &dataSet) override;

                void setOptimizer(T_String);
            };
        }
    }
}
