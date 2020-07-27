#ifndef IMPULSE_NEURALNETWORK_CLASSIFIER_BUILDER_H
#define IMPULSE_NEURALNETWORK_CLASSIFIER_BUILDER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class ClassifierBuilder : public Abstract<Network::ClassifierNetwork> {
            protected:
            public:
                explicit ClassifierBuilder(T_Dimension dims);

                void firstLayerTransition(Layer::LayerPointer layer) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_CLASSIFIER_BUILDER_H
