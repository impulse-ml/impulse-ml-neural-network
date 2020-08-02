#pragma once

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        class ComputationCpu : public AbstractComputation {
        private:
            explicit ComputationCpu();

        public:
            static ComputationCpu factory();

            Eigen::MatrixXd forward(const Eigen::MatrixXd &, const Eigen::MatrixXd &, const Eigen::VectorXd &);

            Eigen::MatrixXd randomInit(Eigen::MatrixXd &, T_Size);

            Eigen::VectorXd randomInit(Eigen::VectorXd &, T_Size);

            Eigen::VectorXd staticInit(Eigen::VectorXd &, double);

            Eigen::MatrixXd reluActivation(Eigen::MatrixXd &);

            Eigen::MatrixXd reluDerivative(Eigen::MatrixXd &);

            Eigen::MatrixXd logisticActivation(Eigen::MatrixXd &);

            Eigen::MatrixXd logisticDerivative(Eigen::MatrixXd &);

            Eigen::MatrixXd softmaxActivation(Eigen::MatrixXd &);

            Eigen::MatrixXd softmaxDerivative(Eigen::MatrixXd &);

            Eigen::MatrixXd softplusActivation(Eigen::MatrixXd &);

            Eigen::MatrixXd softplusDerivative(Eigen::MatrixXd &);

            Eigen::MatrixXd tanhActivation(Eigen::MatrixXd &);

            Eigen::MatrixXd tanhDerivative(Eigen::MatrixXd &);

            double logisticLoss(Eigen::MatrixXd &, Eigen::MatrixXd &);

            double purelinLoss(Eigen::MatrixXd &, Eigen::MatrixXd &);

            double softmaxLoss(Eigen::MatrixXd &, Eigen::MatrixXd &);

            void gradientDescent(Eigen::MatrixXd &, double, Eigen::MatrixXd &);

            void gradientDescent(Eigen::VectorXd &, double, Eigen::VectorXd &);

            double layerPenaltyMiniBatchGradientDescent(Eigen::MatrixXd &);

            void
            gradientAdam(Eigen::MatrixXd &, double, Eigen::MatrixXd &, Eigen::MatrixXd &, Eigen::MatrixXd &, T_Size);

            void
            gradientAdam(Eigen::VectorXd &, double, Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &, T_Size);
        };
    }
}
