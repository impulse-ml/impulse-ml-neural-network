#ifndef IMPULSE_NEURALNETWORK_NETWORK_H
#define IMPULSE_NEURALNETWORK_NETWORK_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            typedef std::vector<Layer::LayerPointer> LayersContainer;

            class Abstract {
            protected:
                T_Size size = 0;
                T_Dimension dimension;
                LayersContainer layers;
            public:
                explicit Abstract(T_Dimension dim);

                void addLayer(Layer::LayerPointer layer);

                Math::T_Matrix forward(const Math::T_Matrix &input);

                void backward(Math::T_Matrix X, Math::T_Matrix Y, Math::T_Matrix predictions, double regularization);

                T_Dimension getDimension();

                T_Size getSize();

                Layer::LayerPointer getLayer(T_Size key);

                Math::T_Vector getRolledTheta();

                Math::T_Vector getRolledGradient();

                void setRolledTheta(Math::T_Vector theta);

                double loss(Math::T_Matrix output, Math::T_Matrix predictions);

                double error(T_Size m);

                void debug();
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_NETWORK_H
