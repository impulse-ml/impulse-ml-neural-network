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

        void ComputationCpu::gradientDescent(Eigen::MatrixXd &W, double learningRate, Eigen::MatrixXd &gW) {
            W = W.array() - learningRate * gW.array();
        }

        void ComputationCpu::gradientDescent(Eigen::VectorXd &b, double learningRate, Eigen::VectorXd &gb) {
            b = b.array() - learningRate * gb.array();
        }

        double ComputationCpu::layerPenaltyMiniBatchGradientDescent(Eigen::MatrixXd &W) {
            return W.unaryExpr([](const double x) {
                return pow(x, 2.0);
            }).sum();
        }

        void
        ComputationCpu::gradientAdam(Eigen::MatrixXd &W, double learningRate, Eigen::MatrixXd &gW, Eigen::MatrixXd &s,
                                     Eigen::MatrixXd &v, T_Size t) {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            v = beta1 * v + (1 - beta1) * gW;
            Eigen::MatrixXd wCorrected = v / (1 - std::pow(beta1, t));

            s = beta2 * s + (1 - beta2) * gW.unaryExpr([](double x) {
                return std::pow(x, 2);
            });

            Eigen::MatrixXd sCorrected = s / (1 - std::pow(beta2, t));
            sCorrected = sCorrected.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            W = W.array() - learningRate * (wCorrected.array() / sCorrected.array());
        }

        void
        ComputationCpu::gradientAdam(Eigen::VectorXd &b, double learningRate, Eigen::VectorXd &gb, Eigen::VectorXd &s,
                                     Eigen::VectorXd &v, T_Size t) {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            v = beta1 * v + (1 - beta1) * gb;
            Eigen::MatrixXd wCorrected = v / (1 - std::pow(beta1, t));

            s = beta2 * s + (1 - beta2) * gb.unaryExpr([](double x) {
                return std::pow(x, 2);
            });

            Eigen::MatrixXd sCorrected = s / (1 - std::pow(beta2, t));
            sCorrected = sCorrected.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            b = b.array() - learningRate * (wCorrected.array() / sCorrected.array());
        }

        void ComputationCpu::gradientRmsProp(Eigen::MatrixXd & W, double learningRate, Eigen::MatrixXd & gW, Eigen::MatrixXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;
            double epsilon = 1e-8;

            s = gamma * s + (1.0 - gamma) * W.unaryExpr([](double x) {
                return std::pow(x, 2);
            });
            W = W.array() - (alpha * gW.array()) / s.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array();
        }

        void ComputationCpu::gradientRmsProp(Eigen::VectorXd & b, double learningRate, Eigen::VectorXd & gb, Eigen::VectorXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;
            double epsilon = 1e-8;

            s = gamma * s + (1.0 - gamma) * b.unaryExpr([](double x) {
                return std::pow(x, 2);
            });
            b = b.array() - (alpha * gb.array()) / s.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array();
        }

        void ComputationCpu::gradientAdagrad(Eigen::MatrixXd & W, double learningRate, Eigen::MatrixXd & gW, Eigen::MatrixXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double epsilon = 1e-8;

            s = s.array() + s.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            W = W.array() - (alpha * gW.array() / s.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());
        }

        void ComputationCpu::gradientAdagrad(Eigen::VectorXd & b, double learningRate, Eigen::VectorXd & gb, Eigen::VectorXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double epsilon = 1e-8;

            s = s.array() + s.unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            b = b.array() - (alpha * gb.array() / s.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array());
        }

        void ComputationCpu::gradientNesterov(Eigen::MatrixXd & W, double learningRate, Eigen::MatrixXd & gW, Eigen::MatrixXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            Eigen::MatrixXd s_prev = s;

            s = (gamma * s.array()) + (alpha * gW.array());
            W = W.array() - ((1.0 + gamma) * s.array() - gamma * s_prev.array());
        }

        void ComputationCpu::gradientNesterov(Eigen::VectorXd & b, double learningRate, Eigen::VectorXd & gb, Eigen::VectorXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            Eigen::MatrixXd s_prev = s;

            s = (gamma * s.array()) + (alpha * gb.array());
            b = b.array() - ((1.0 + gamma) * s.array() - gamma * s_prev.array());
        }

        void ComputationCpu::gradientMomentum(Eigen::MatrixXd & W, double learningRate, Eigen::MatrixXd & gW, Eigen::MatrixXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            s = (gamma * s.array()) + (alpha * gW.array());
            W = W.array() - s.array();
        }

        void ComputationCpu::gradientMomentum(Eigen::VectorXd & b, double learningRate, Eigen::VectorXd & gb, Eigen::VectorXd & s, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            s = (gamma * s.array()) + (alpha * gb.array());
            b = b.array() - s.array();
        }
    }
}
