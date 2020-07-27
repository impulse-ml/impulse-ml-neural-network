#ifndef IMPULSE_NEURALNETWORK_LAYER_FULLYCONNECTED_H
#define IMPULSE_NEURALNETWORK_LAYER_FULLYCONNECTED_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_FULLYCONNECTED = "fully-connected";

            class FullyConnected : public Conv {
            protected:
            public:
                FullyConnected();

                const T_String getType() override;

                void transition(Layer::LayerPointer prevLayer) override;

                void setSize(T_Size value) override;

                void setFilterSize(T_Size value) override;

                void setPadding(T_Size value) override;

                void setStride(T_Size value) override;

                void setWidth(T_Size value) override;

                void setHeight(T_Size value) override;

                void setDepth(T_Size value) override;

                void setNumFilters(T_Size value) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_FULLYCONNECTED_H
