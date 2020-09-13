#pragma once

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        class Computation : public AbstractComputation {

        private:
            ComputationCpu *computation;

        public:
            explicit Computation();

            Eigen::MatrixXd &getVariable(T_String);

            void initialize(T_String);

            void resize(T_String, T_Size, T_Size);

            void setZero(T_String);

            void randomInit(T_String, double);

            void setVariable(T_String, Eigen::MatrixXd);

            Eigen::MatrixXd forward(const Eigen::MatrixXd &);

            void reluActivation();

            Eigen::MatrixXd reluDerivative(Eigen::MatrixXd &);

            void logisticActivation();

            Eigen::MatrixXd logisticDerivative(Eigen::MatrixXd &);

            void softmaxActivation();

            Eigen::MatrixXd softmaxDerivative(Eigen::MatrixXd &);

            void softplusActivation();

            Eigen::MatrixXd softplusDerivative(Eigen::MatrixXd &);

            void tanhActivation();

            Eigen::MatrixXd tanhDerivative(Eigen::MatrixXd &);

            double logisticLoss(Eigen::MatrixXd &, Eigen::MatrixXd &);

            double purelinLoss(Eigen::MatrixXd &, Eigen::MatrixXd &);

            double softmaxLoss(Eigen::MatrixXd &, Eigen::MatrixXd &);

            double penalty();

            void gradientDescent(double);

            void gradientAdam(double, T_Size);

            void gradientRmsProp(double, T_Size);

            void gradientAdagrad(double, T_Size);

            void gradientNesterov(double, T_Size);

            void gradientMomentum(double, T_Size);

            void gradientAdadelta(double, T_Size);
        };
    }
}
