#pragma once

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

                static ClassifierBuilder fromJSON(T_String path);
            };
        }
    }
}
