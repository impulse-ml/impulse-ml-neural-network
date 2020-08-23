#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_CONV = "conv";

            class Conv : public Abstract3D {
            protected:
                T_Size filterSize = 3;
                T_Size padding = 1;
                T_Size stride = 2;
                T_Size numFilters = 2;
            public:
                Conv();

                void configure() override;

                Eigen::MatrixXd forward(const Eigen::MatrixXd &input) override;

                T_Size getOutputHeight() override;

                T_Size getOutputWidth() override;

                T_Size getOutputDepth() override;

                virtual void setFilterSize(T_Size value);

                T_Size getFilterSize();

                virtual void setPadding(T_Size value);

                T_Size getPadding();

                virtual void setStride(T_Size value);

                T_Size getStride();

                virtual void setNumFilters(T_Size value);

                T_Size getNumFilters();

                Eigen::MatrixXd activation() override;

                Eigen::MatrixXd derivative(Eigen::MatrixXd &a) override;

                const T_String getType() override;

                double loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) override;

                double error(T_Size m) override;
            };
        }
    }
}
