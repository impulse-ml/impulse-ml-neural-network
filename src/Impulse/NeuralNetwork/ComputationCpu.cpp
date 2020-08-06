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

        double ComputationCpu::layerPenalty(Eigen::MatrixXd & W) {
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

            layer->sW = beta2 * layer->sW + (1 - beta2) * layer->gW.unaryExpr([](double x) {
                return std::pow(x, 2);
            });

            Eigen::MatrixXd sCorrected = layer->sW / (1 - std::pow(beta2, t));
            sCorrected = sCorrected.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            layer->W = layer->W.array() - learningRate * (wCorrected.array() / sCorrected.array());

            //

            layer->vB = beta1 * layer->vB + (1 - beta1) * layer->gB;
            Eigen::VectorXd wCorrected2 = layer->vB / (1 - std::pow(beta1, t));

            layer->sB = beta2 * layer->sB + (1 - beta2) * layer->gB.unaryExpr([](double x) {
                return std::pow(x, 2);
            });

            Eigen::VectorXd sCorrected2 = layer->sB / (1 - std::pow(beta2, t));
            sCorrected2 = sCorrected2.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            layer->b = layer->b.array() - learningRate * (wCorrected2.array() / sCorrected2.array());
        }

        void ComputationCpu::gradientRmsProp(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;
            double epsilon = 1e-8;

            layer->sW = gamma * layer->sW + (1.0 - gamma) * layer->W.unaryExpr([](double x) {
                return std::pow(x, 2);
            });
            layer->W = layer->W.array() - (alpha * layer->gW.array()) / layer->sW.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array();

            layer->sB = gamma * layer->sB + (1.0 - gamma) * layer->b.unaryExpr([](double x) {
                return std::pow(x, 2);
            });
            layer->b = layer->b.array() - (alpha * layer->gB.array()) / layer->sB.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array();
        }

        void ComputationCpu::gradientAdagrad(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double epsilon = 1e-8;

            layer->sW = layer->sW.array() + layer->gW.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            layer->W = layer->W.array() - (alpha * layer->gW.array() / layer->sW.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());

            layer->sB = layer->sB.array() + layer->gB.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            layer->b = layer->b.array() - (alpha * layer->gB.array() / layer->sB.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());
        }

        void ComputationCpu::gradientNesterov(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            Eigen::MatrixXd s_prev = layer->sW;

            layer->sW = (gamma * layer->sW.array()) + (alpha * layer->gW.array());
            layer->W = layer->W.array() - ((1.0 + gamma) * layer->sW.array() - gamma * s_prev.array());

            //

            Eigen::VectorXd s_prev_b = layer->sB;

            layer->sB = (gamma * layer->sB.array()) + (alpha * layer->gB.array());
            layer->b = layer->b.array() - ((1.0 + gamma) * layer->sB.array() - gamma * s_prev_b.array());
        }

        void ComputationCpu::gradientMomentum(Layer::Abstract *layer, double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            layer->sW = (gamma * layer->sW.array()) + (alpha * layer->gW.array());
            layer->W = layer->W.array() - layer->sW.array();

            layer->sB = (gamma * layer->sB.array()) + (alpha * layer->gB.array());
            layer->b = layer->b.array() - layer->sB.array();
        }

    }
}
