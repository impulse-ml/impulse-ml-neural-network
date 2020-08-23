#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_LOGISTIC = "logistic";

            class Logistic : public Abstract1D {
            protected:
            public:
                Logistic();

                Eigen::MatrixXd activation() override;

                Eigen::MatrixXd derivative(Eigen::MatrixXd &a) override;

                const T_String getType() override;

                double loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) override;

                double error(T_Size m) override;
            };
        }
    }
}
