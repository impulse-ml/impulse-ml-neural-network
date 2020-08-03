#pragma once

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        class Computation : public AbstractComputation {
        private:
            explicit Computation();

        public:
            static Computation factory();

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

            void gradientRmsProp(Eigen::MatrixXd &, double, Eigen::MatrixXd &, Eigen::MatrixXd &, T_Size);

            void gradientRmsProp(Eigen::VectorXd &, double, Eigen::VectorXd &, Eigen::VectorXd &, T_Size);

            void gradientAdagrad(Eigen::MatrixXd &, double, Eigen::MatrixXd &, Eigen::MatrixXd &, T_Size);

            void gradientAdagrad(Eigen::VectorXd &, double, Eigen::VectorXd &, Eigen::VectorXd &, T_Size);

            void gradientNesterov(Eigen::MatrixXd &, double, Eigen::MatrixXd &, Eigen::MatrixXd &, T_Size);

            void gradientNesterov(Eigen::VectorXd &, double, Eigen::VectorXd &, Eigen::VectorXd &, T_Size);

            void gradientMomentum(Eigen::MatrixXd &, double, Eigen::MatrixXd &, Eigen::MatrixXd &, T_Size);

            void gradientMomentum(Eigen::VectorXd &, double, Eigen::VectorXd &, Eigen::VectorXd &, T_Size);
        };
    }
}
