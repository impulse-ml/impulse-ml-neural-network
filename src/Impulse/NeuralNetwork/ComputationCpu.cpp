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
                return 1.0 / (1.0 + std::exp(-x));
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

        Eigen::MatrixXd ComputationCpu::softplusActivation(Eigen::MatrixXd &m) {
            Eigen::MatrixXd result = m.unaryExpr([](const double x) {
                return std::log(1.0 + std::exp(x));
            });
            return result;
        }

        Eigen::MatrixXd ComputationCpu::softplusDerivative(Eigen::MatrixXd &m) {
            Eigen::MatrixXd result = m.unaryExpr([](const double x) {
                return (1.0 / (1.0 + std::exp(-x)));
            });
            return result;
        }

        Eigen::MatrixXd ComputationCpu::tanhActivation(Eigen::MatrixXd &m) {
            Eigen::MatrixXd result = m.unaryExpr([](const double x) {
                return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1;
            });
            return result;
        }

        Eigen::MatrixXd ComputationCpu::tanhDerivative(Eigen::MatrixXd &m) {
            Eigen::MatrixXd result = m.unaryExpr([](const double x) {
                return 1.0 - std::pow((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0, 2);
            });
            return result;
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

        void ComputationCpu::gradientDescent(Layer::Abstract *layer, double learningRate) {
            layer->W = layer->W.array() - learningRate * layer->gW.array();
            layer->b = layer->b.array() - learningRate * layer->gB.array();
        }

        double ComputationCpu::layerPenalty(Eigen::MatrixXd &W) {
            return W.unaryExpr([](const double x) {
                return pow(x, 2.0);
            }).sum();
        }

        void
        ComputationCpu::gradientAdam(Layer::Abstract *layer, double learningRate, T_Size t) {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            layer->vW = beta1 * layer->vW + (1 - beta1) * layer->gW;
            Eigen::MatrixXd wCorrected = layer->vW / (1 - std::pow(beta1, t));

            layer->cW = beta2 * layer->cW.array() + (1 - beta2) * (layer->gW.array() * layer->gW.array());

            Eigen::MatrixXd sCorrected = layer->cW / (1 - std::pow(beta2, t));
            sCorrected = sCorrected.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            layer->W = layer->W.array() - learningRate * (wCorrected.array() / sCorrected.array());

            //

            layer->vB = beta1 * layer->vB + (1 - beta1) * layer->gB;
            Eigen::VectorXd wCorrected2 = layer->vB / (1 - std::pow(beta1, t));

            layer->cB = beta2 * layer->cB.array() + (1 - beta2) * (layer->gB.array() * layer->gB.array());

            Eigen::VectorXd sCorrected2 = layer->cB / (1 - std::pow(beta2, t));
            sCorrected2 = sCorrected2.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            layer->b = layer->b.array() - learningRate * (wCorrected2.array() / sCorrected2.array());
        }

        void ComputationCpu::gradientRmsProp(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double alpha = 1e-3;
            double gamma = 0.9;
            double epsilon = 1e-8;

            layer->cW = gamma * layer->cW.array() + (1.0 - gamma) * (layer->gW.array() * layer->gW.array());
            layer->W = layer->W.array() - (layer->gW.array() * alpha / layer->cW.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());

            layer->cB = gamma * layer->cB.array() + (1.0 - gamma) * (layer->gB.array() * layer->gB.array());
            layer->b = layer->b.array() - (layer->gB.array() * alpha / layer->cB.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());
        }

        void ComputationCpu::gradientAdagrad(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double epsilon = 1e-8;

            layer->cW = layer->cW.array() + layer->gW.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            layer->W = layer->W.array() - (alpha * layer->gW.array() / layer->cW.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());

            layer->cB = layer->cB.array() + layer->gB.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            layer->b = layer->b.array() - (alpha * layer->gB.array() / layer->cB.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());
        }

        void ComputationCpu::gradientNesterov(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double gamma = 0.9;

            Eigen::MatrixXd s_prev = layer->cW;

            layer->cW = (gamma * layer->cW.array()) - (learningRate * layer->gW.array());
            layer->W = layer->W.array() + layer->cW.array() + (gamma * (layer->cW.array() - s_prev.array()));


            Eigen::VectorXd s_prev_b = layer->cB;

            layer->cB = (gamma * layer->cB.array()) - (learningRate * layer->gB.array());
            layer->b = layer->b.array() + layer->cB.array() + (gamma * (layer->cB.array() - s_prev_b.array()));
        }

        void ComputationCpu::gradientMomentum(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            layer->cW = (gamma * layer->cW.array()) + (alpha * layer->gW.array());
            layer->W = layer->W.array() - layer->cW.array();

            layer->cB = (gamma * layer->cB.array()) + (alpha * layer->gB.array());
            layer->b = layer->b.array() - layer->cB.array();
        }

        void ComputationCpu::gradientAdadelta(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            //double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;
            double epsilon = 1e-6;

            layer->cW = (gamma * layer->cW.array()) + (1.0 - gamma) * (layer->gW.array() * layer->gW.array());
            Eigen::MatrixXd deltaParameters = - (layer->vW.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array() / layer->cW.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array()).array() * layer->gW.array();
            layer->vW = (gamma * layer->cW.array()) + ((1.0 - gamma) * (deltaParameters.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array()));
            layer->W = layer->W.array() + deltaParameters.array();

            layer->cB = (gamma * layer->cB.array()) + (1.0 - gamma) * (layer->gB.array() * layer->gB.array());
            Eigen::MatrixXd deltaParameters2 = - (layer->vB.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array() / layer->cB.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array()) * layer->gB.array();
            layer->vB = (gamma * layer->cB.array()) + ((1.0 - gamma) * (deltaParameters2.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array()));
            layer->b = layer->b.array() + deltaParameters2.array();
        }
    }
}
