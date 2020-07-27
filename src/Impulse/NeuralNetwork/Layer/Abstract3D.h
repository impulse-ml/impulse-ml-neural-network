#ifndef IMPULSE_NEURALNETWORK_LAYER_3D_H
#define IMPULSE_NEURALNETWORK_LAYER_3D_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract3D : public Abstract {
            public:
                Abstract3D();

                bool is1D() override;

                bool is3D() override;

                void transition(Layer::LayerPointer prevLayer) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_3D_H
