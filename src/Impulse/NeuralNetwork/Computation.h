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

            void gradientDescent(Layer::Abstract *, double);

            double layerPenalty(Eigen::MatrixXd &W);

            void gradientAdam(Layer::Abstract *, double, T_Size);

            void gradientRmsProp(Layer::Abstract *, double, T_Size);

            void gradientAdagrad(Layer::Abstract *, double, T_Size);

            void gradientNesterov(Layer::Abstract *, double, T_Size);

            void gradientMomentum(Layer::Abstract *, double, T_Size);
        };
    }
}
