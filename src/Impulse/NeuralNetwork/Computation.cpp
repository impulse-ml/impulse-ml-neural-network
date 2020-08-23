#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Computation::Computation() : AbstractComputation() {
            this->computation = new ComputationCpu();
        }

        Eigen::MatrixXd &Computation::getVariable(T_String name) {
            return this->computation->getVariable(name);
        }

        void Computation::setVariable(T_String name, Eigen::MatrixXd variable) {
            this->computation->setVariable(name, variable);
        }

        void Computation::initialize(T_String name) {
            this->computation->initialize(name);
        }

        Eigen::MatrixXd
        Computation::forward(const Eigen::MatrixXd &input) {
            return this->computation->forward(input);
        }

        void Computation::resize(T_String name, T_Size width, T_Size height) {
            this->computation->resize(name, width, height);
        }

        void Computation::setZero(T_String name) {
            this->computation->setZero(name);
        }

        void Computation::randomInit(T_String name, double parameter) {
            this->computation->randomInit(name, parameter);
        }

        void Computation::reluActivation() {
            return this->computation->reluActivation();
        }

        Eigen::MatrixXd Computation::reluDerivative(Eigen::MatrixXd &m) {
            return this->computation->reluDerivative(m);
        }

        void Computation::logisticActivation() {
            return this->computation->logisticActivation();
        }

        Eigen::MatrixXd Computation::logisticDerivative(Eigen::MatrixXd &m) {
            return this->computation->logisticDerivative(m);
        }

        void Computation::softmaxActivation() {
            return this->computation->softmaxActivation();
        }

        Eigen::MatrixXd Computation::softmaxDerivative(Eigen::MatrixXd &m) {
            return this->computation->softmaxDerivative(m);
        }

        void Computation::softplusActivation() {
            return this->computation->softplusActivation();
        }

        Eigen::MatrixXd Computation::softplusDerivative(Eigen::MatrixXd &m) {
            return this->computation->softplusDerivative(m);
        }

        void Computation::tanhActivation() {
            return this->computation->tanhActivation();
        }

        Eigen::MatrixXd Computation::tanhDerivative(Eigen::MatrixXd &m) {
            return this->computation->tanhDerivative(m);
        }

        double Computation::logisticLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            return this->computation->logisticLoss(output, predictions);
        }

        double Computation::purelinLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            return this->computation->purelinLoss(output, predictions);
        }

        double Computation::softmaxLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            return this->computation->softmaxLoss(output, predictions);
        }

        double Computation::penalty() {
            return this->computation->penalty();
        }

        void Computation::gradientDescent(double learningRate) {
            this->computation->gradientDescent(learningRate);
        }

        void Computation::gradientAdam(double learningRate, T_Size t) {
            this->computation->gradientAdam(learningRate, t);
        }

        void Computation::gradientRmsProp(double learningRate, T_Size batchSize) {
            this->computation->gradientRmsProp(learningRate, batchSize);
        }

        void Computation::gradientAdagrad(double learningRate, T_Size batchSize) {
            this->computation->gradientAdagrad(learningRate, batchSize);
        }

        void Computation::gradientNesterov(double learningRate, T_Size batchSize) {
            this->computation->gradientNesterov(learningRate, batchSize);
        }

        void Computation::gradientMomentum(double learningRate, T_Size batchSize) {
            this->computation->gradientMomentum(learningRate, batchSize);
        }

        void Computation::gradientAdadelta(double learningRate, T_Size batchSize) {
            this->computation->gradientAdadelta(learningRate, batchSize);
        }
    }
}
