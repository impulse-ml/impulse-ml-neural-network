#pragma once

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

                Eigen::MatrixXd forward(const Eigen::MatrixXd & input);

                void backward(Eigen::MatrixXd & X, Eigen::MatrixXd & Y, Eigen::MatrixXd & predictions, double regularization);

                T_Dimension getDimension();

                T_Size getSize();

                Layer::LayerPointer getLayer(T_Size key);

                Eigen::VectorXd getRolledTheta();

                Eigen::VectorXd getRolledGradient();

                void setRolledTheta(Eigen::VectorXd & theta);

                double loss(Eigen::MatrixXd & output, Eigen::MatrixXd & predictions);

                double error(T_Size m);

                void debug();
            };
        }
    }
}
