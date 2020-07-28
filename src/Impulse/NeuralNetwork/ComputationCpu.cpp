#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        ComputationCpu::ComputationCpu() : AbstractComputation() {

        }

        ComputationCpu ComputationCpu::factory() {
            ComputationCpu instance;
            return instance;
        }

        Eigen::MatrixXd
        ComputationCpu::forward(const Eigen::MatrixXd &W, const Eigen::MatrixXd &input, const Eigen::VectorXd &b) {
            return (W * input).colwise() + b;
        }

        Eigen::MatrixXd ComputationCpu::randomInit(Eigen::MatrixXd &mat, T_Size width) {
            mat.setRandom();
            return mat.unaryExpr([width](const double x) {
                return x * sqrt(2.0 / width);
            });
        }

        Eigen::VectorXd ComputationCpu::randomInit(Eigen::VectorXd &vec, T_Size width) {
            vec.setRandom();
            return vec.unaryExpr([width](const double x) {
                return x * sqrt(2.0 / width);
            });
        }

        Eigen::VectorXd ComputationCpu::staticInit(Eigen::VectorXd &vec, double num) {
            vec.setRandom();
            return vec.unaryExpr([num](const double x) {
                return num;
            });
        }

        Eigen::MatrixXd ComputationCpu::reluActivation(Eigen::MatrixXd &m) {
            return m.unaryExpr([](const double x) {
                return std::max(0.0, x);
            });
        }

        Eigen::MatrixXd ComputationCpu::reluDerivative(Eigen::MatrixXd &m) {
            return m.unaryExpr([](const double x) {
                if (x > 0.0) {
                    return 1.0;
                }
                return 0.0;
            });
        }

        Eigen::MatrixXd ComputationCpu::logisticActivation(Eigen::MatrixXd &m) {
            return m.unaryExpr([](const double x) {
                return 1.0 / (1.0 + exp(-x));
            });
        }

        Eigen::MatrixXd ComputationCpu::logisticDerivative(Eigen::MatrixXd &m) {
            return m.array() * (1.0 - m.array());
        }

        Eigen::MatrixXd ComputationCpu::softmaxActivation(Eigen::MatrixXd &m) {
            Eigen::MatrixXd t = m.unaryExpr([](const double x) {
                return exp(x);
            });
            Eigen::MatrixXd divider = t.colwise().sum().replicate(t.rows(), 1);
            Eigen::MatrixXd result = t.array() / divider.array();
            return result;
        }

        Eigen::MatrixXd ComputationCpu::softmaxDerivative(Eigen::MatrixXd &) {
            return Eigen::MatrixXd(); // TODO
        }

        double ComputationCpu::logisticLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            Eigen::MatrixXd loss =
                    (output.array() * predictions.unaryExpr([](const double x) { return log(x); }).array())
                    +
                    (output.unaryExpr([](const double x) { return 1.0 - x; }).array()
                     *
                     predictions.unaryExpr([](const double x) { return log(1.0 - x); }).array()
                    );
            return loss.sum();
        }

        double ComputationCpu::purelinLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            Eigen::MatrixXd loss = (predictions.array() - output.array()).unaryExpr([](const double x) {
                return pow(x, 2.0);
            });
            return loss.sum();
        }

        double ComputationCpu::softmaxLoss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
            Eigen::MatrixXd loss = (output.array() *
                                    predictions.unaryExpr([](const double x) { return log(x); }).array());
            return loss.sum();
        }

        Eigen::MatrixXd ComputationCpu::gradientDescent(Eigen::MatrixXd &W, double learningRate, Eigen::MatrixXd &gW) {
            return W.array() - learningRate * gW.array();
        }

        Eigen::VectorXd ComputationCpu::gradientDescent(Eigen::VectorXd &b, double learningRate, Eigen::VectorXd &gb) {
            return b.array() - learningRate * gb.array();
        }

        double ComputationCpu::layerPenaltyMiniBatchGradientDescent(Eigen::MatrixXd &W) {
            return W.unaryExpr([](const double x) {
                return pow(x, 2.0);
            }).sum();
        }
    }
}
