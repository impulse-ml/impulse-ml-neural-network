#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_MAXPOOL = "maxpool";

            class MaxPool : public Abstract3D {
            protected:
                T_Size filterSize = 2;
                T_Size stride = 2;
            public:
                MaxPool();

                void configure() override;

                void setFilterSize(T_Size value);

                T_Size getFilterSize();

                void setStride(T_Size value);

                T_Size getStride();

                Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

                Eigen::MatrixXd activation() override;

                Eigen::MatrixXd derivative(Eigen::MatrixXd &a) override;

                const T_String getType() override;

                double loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) override;

                double error(T_Size m) override;

                T_Size getOutputHeight() override;

                T_Size getOutputWidth() override;

                T_Size getOutputDepth() override;
            };
        }
    }
}
