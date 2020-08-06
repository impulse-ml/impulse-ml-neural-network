#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Computation::Computation() : AbstractComputation() {

        }

        Computation Computation::factory() {
            Computation instance;
            return instance;
        }

        Eigen::MatrixXd
        Computation::forward(const Eigen::MatrixXd &W, const Eigen::MatrixXd &input, const Eigen::VectorXd &b) {
            return ComputationCpu::factory().forward(W, input, b);
        }

        Eigen::MatrixXd Computation::randomInit(Eigen::MatrixXd &mat, T_Size width) {
            return ComputationCpu::factory().randomInit(mat, width);
        }

        Eigen::VectorXd Computation::randomInit(Eigen::VectorXd &vec, T_Size width) {
            return ComputationCpu::factory().randomInit(vec, width);
        }

        Eigen::VectorXd Computation::staticInit(Eigen::VectorXd &vec, double num) {
            return ComputationCpu::factory().staticInit(vec, num);
        }

        Eigen::MatrixXd Computation::reluActivation(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().reluActivation(m);
        }

        Eigen::MatrixXd Computation::reluDerivative(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().reluDerivative(m);
        }

        Eigen::MatrixXd Computation::logisticActivation(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().logisticActivation(m);
        }

        Eigen::MatrixXd Computation::logisticDerivative(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().logisticDerivative(m);
        }

        Eigen::MatrixXd Computation::softmaxActivation(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().softmaxActivation(m);
        }

        Eigen::MatrixXd Computation::softmaxDerivative(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().softmaxDerivative(m);
        }

        Eigen::MatrixXd Computation::softplusActivation(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().softplusActivation(m);
        }

        Eigen::MatrixXd Computation::softplusDerivative(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().softplusDerivative(m);
        }

        Eigen::MatrixXd Computation::tanhActivation(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().tanhActivation(m);
        }

        Eigen::MatrixXd Computation::tanhDerivative(Eigen::MatrixXd &m) {
            return ComputationCpu::factory().tanhDerivative(m);
        }

        double Computation::logisticLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            return ComputationCpu::factory().logisticLoss(output, predictions);
        }

        double Computation::purelinLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            return ComputationCpu::factory().purelinLoss(output, predictions);
        }

        double Computation::softmaxLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            return ComputationCpu::factory().softmaxLoss(output, predictions);
        }

        void Computation::gradientDescent(Layer::Abstract *layer, double learningRate) {
            ComputationCpu::factory().gradientDescent(layer, learningRate);
        }

        double Computation::layerPenalty(Eigen::MatrixXd &W) {
            return ComputationCpu::factory().layerPenalty(W);
        }

        void Computation::gradientAdam(Layer::Abstract *layer, double learningRate, T_Size t) {
            ComputationCpu::factory().gradientAdam(layer, learningRate, t);
        }

        void Computation::gradientRmsProp(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            ComputationCpu::factory().gradientRmsProp(layer, learningRate, batchSize);
        }

        void Computation::gradientAdagrad(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            ComputationCpu::factory().gradientAdagrad(layer, learningRate, batchSize);
        }

        void Computation::gradientNesterov(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            ComputationCpu::factory().gradientNesterov(layer, learningRate, batchSize);
        }

        void Computation::gradientMomentum(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            ComputationCpu::factory().gradientMomentum(layer, learningRate, batchSize);
        }
    }
}
