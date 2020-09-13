#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        ComputationCpu::ComputationCpu() : AbstractComputation() {

        }

        void ComputationCpu::initialize(T_String name) {
            this->variables[name] = Eigen::MatrixXd();
        }

        void ComputationCpu::resize(T_String name, T_Size width, T_Size height) {
            this->variables[name].resize(width, height);
        }

        void ComputationCpu::setVariable(T_String name, Eigen::MatrixXd variable) {
            this->variables[name] = variable;
        }

        Eigen::MatrixXd &ComputationCpu::getVariable(T_String name) {
            return this->variables[name];
        }

        void ComputationCpu::setZero(T_String name) {
            this->variables[name].setZero();
        }

        void ComputationCpu::randomInit(T_String name, double parameter) {
            this->variables[name].setRandom();
            this->variables[name] = this->variables[name].unaryExpr([parameter](const double x) {
                return x * sqrt(2.0 / parameter);
            });
        }

        Eigen::MatrixXd
        ComputationCpu::forward(const Eigen::MatrixXd &input) {
            this->setVariable("Z", (this->variables["W"] * input) + this->variables["b"].replicate(1, input.cols()));
            return this->getVariable("Z");
        }

        void ComputationCpu::reluActivation() {
            this->variables["A"] = this->variables["Z"].unaryExpr([](const double x) {
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

        void ComputationCpu::logisticActivation() {
            this->variables["A"] = this->variables["Z"].unaryExpr([](const double x) {
                return 1.0 / (1.0 + std::exp(-x));
            });
        }

        Eigen::MatrixXd ComputationCpu::logisticDerivative(Eigen::MatrixXd &m) {
            return m.array() * (1.0 - m.array());
        }

        void ComputationCpu::softmaxActivation() {
            Eigen::MatrixXd t = this->variables["Z"].unaryExpr([](const double x) {
                return exp(x);
            });
            Eigen::MatrixXd divider = t.colwise().sum().replicate(t.rows(), 1);
            Eigen::MatrixXd result = t.array() / divider.array();

            this->variables["A"] = result;
        }

        Eigen::MatrixXd ComputationCpu::softmaxDerivative(Eigen::MatrixXd &) {
            return Eigen::MatrixXd(); // TODO
        }

        void ComputationCpu::softplusActivation() {
            this->variables["A"] = this->variables["Z"].unaryExpr([](const double x) {
                return std::log(1.0 + std::exp(x));
            });
        }

        Eigen::MatrixXd ComputationCpu::softplusDerivative(Eigen::MatrixXd &m) {
            Eigen::MatrixXd result = m.unaryExpr([](const double x) {
                return (1.0 / (1.0 + std::exp(-x)));
            });
            return result;
        }

        void ComputationCpu::tanhActivation() {
            this->variables["A"] = this->variables["Z"].unaryExpr([](const double x) {
                return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1;
            });
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

        void ComputationCpu::gradientDescent(double learningRate) {
            this->variables["W"] = this->variables["W"].array() - learningRate * this->variables["gW"].array();
            this->variables["b"] = this->variables["b"].array() - learningRate * this->variables["gB"].array();
        }

        double ComputationCpu::penalty() {
            return this->variables["W"].unaryExpr([](const double x) {
                return pow(x, 2.0);
            }).sum();
        }

        void
        ComputationCpu::gradientAdam(double learningRate, T_Size t) {
            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = 1e-8;

            this->variables["vW"] = beta1 * this->variables["vW"] + (1 - beta1) * this->variables["gW"];
            Eigen::MatrixXd wCorrected = this->variables["vW"] / (1 - std::pow(beta1, t));

            this->variables["cW"] = beta2 * this->variables["cW"].array() +
                                    (1 - beta2) * (this->variables["gW"].array() * this->variables["gW"].array());

            Eigen::MatrixXd sCorrected = this->variables["cW"] / (1 - std::pow(beta2, t));
            sCorrected = sCorrected.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            this->variables["W"] =
                    this->variables["W"].array() - learningRate * (wCorrected.array() / sCorrected.array());

            //

            this->variables["vB"] = beta1 * this->variables["vB"] + (1 - beta1) * this->variables["gB"];
            Eigen::VectorXd wCorrected2 = this->variables["vB"] / (1 - std::pow(beta1, t));

            this->variables["cB"] = beta2 * this->variables["cB"].array() +
                                    (1 - beta2) * (this->variables["gB"].array() * this->variables["gB"].array());

            Eigen::VectorXd sCorrected2 = this->variables["cB"] / (1 - std::pow(beta2, t));
            sCorrected2 = sCorrected2.unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            });

            this->variables["b"] =
                    this->variables["b"].array() - learningRate * (wCorrected2.array() / sCorrected2.array());
        }

        void ComputationCpu::gradientRmsProp(double learningRate, T_Size batchSize) {
            double alpha = 1e-3;
            double gamma = 0.9;
            double epsilon = 1e-8;

            this->variables["cW"] = gamma * this->variables["cW"].array() +
                                    (1.0 - gamma) * (this->variables["gW"].array() * this->variables["gW"].array());
            this->variables["W"] = this->variables["W"].array() - (this->variables["gW"].array() * alpha /
                                                                   this->variables["cW"].unaryExpr([epsilon](double x) {
                                                                       return std::sqrt(x + epsilon);
                                                                   }).array());

            this->variables["cB"] = gamma * this->variables["cB"].array() +
                                    (1.0 - gamma) * (this->variables["gB"].array() * this->variables["gB"].array());
            this->variables["b"] = this->variables["b"].array() - (this->variables["gB"].array() * alpha /
                                                                   this->variables["cB"].unaryExpr([epsilon](double x) {
                                                                       return std::sqrt(x + epsilon);
                                                                   }).array());
        }

        void ComputationCpu::gradientAdagrad(double learningRate, T_Size batchSize) {
            double epsilon = 1e-8;

            this->variables["cW"] = this->variables["cW"].array() + this->variables["gW"].unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            this->variables["W"] = this->variables["W"].array() - (learningRate * this->variables["gW"].array() /
                                                                   this->variables["cW"].unaryExpr([epsilon](double x) {
                                                                       return std::sqrt(x + epsilon);
                                                                   }).array());

            this->variables["cB"] = this->variables["cB"].array() + this->variables["gB"].unaryExpr([](double x) {
                return std::pow(x, 2);
            }).array();
            this->variables["b"] = this->variables["b"].array() - (learningRate * this->variables["gB"].array() /
                                                                   this->variables["cB"].unaryExpr([epsilon](double x) {
                                                                       return std::sqrt(x + epsilon);
                                                                   }).array());
        }

        void ComputationCpu::gradientNesterov(double learningRate, T_Size batchSize) {
            double gamma = 0.9;

            Eigen::MatrixXd s_prev = this->variables["cW"];

            this->variables["cW"] =
                    (gamma * this->variables["cW"].array()) - (learningRate * this->variables["gW"].array());
            this->variables["W"] = this->variables["W"].array() + this->variables["cW"].array() +
                                   (gamma * (this->variables["cW"].array() - s_prev.array()));


            Eigen::VectorXd s_prev_b = this->variables["cB"];

            this->variables["cB"] =
                    (gamma * this->variables["cB"].array()) - (learningRate * this->variables["gB"].array());
            this->variables["b"] = this->variables["b"].array() + this->variables["cB"].array() +
                                   (gamma * (this->variables["cB"].array() - s_prev_b.array()));
        }

        void ComputationCpu::gradientMomentum(double learningRate, T_Size batchSize) {
            double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;

            this->variables["cW"] = (gamma * this->variables["cW"].array()) + (alpha * this->variables["gW"].array());
            this->variables["W"] = this->variables["W"].array() - this->variables["cW"].array();

            this->variables["cB"] = (gamma * this->variables["cB"].array()) + (alpha * this->variables["gB"].array());
            this->variables["b"] = this->variables["b"].array() - this->variables["cB"].array();
        }

        void ComputationCpu::gradientAdadelta(double learningRate, T_Size batchSize) {
            //double alpha = learningRate / (double) batchSize;
            double gamma = 0.9;
            double epsilon = 1e-6;

            this->variables["cW"] = (gamma * this->variables["cW"].array()) +
                                    (1.0 - gamma) * (this->variables["gW"].array() * this->variables["gW"].array());
            Eigen::MatrixXd deltaParameters = -(this->variables["vW"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array() / this->variables["cW"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array()).array() * this->variables["gW"].array();
            this->variables["vW"] =
                    (gamma * this->variables["cW"].array()) + ((1.0 - gamma) * (deltaParameters.unaryExpr([](double x) {
                        return std::pow(x, 2);
                    }).array()));
            this->variables["W"] = this->variables["W"].array() + deltaParameters.array();

            this->variables["cB"] = (gamma * this->variables["cB"].array()) +
                                    (1.0 - gamma) * (this->variables["gB"].array() * this->variables["gB"].array());
            Eigen::MatrixXd deltaParameters2 = -(this->variables["vB"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array() / this->variables["cB"].unaryExpr([epsilon](double x) {
                return std::sqrt(x + epsilon);
            }).array()) * this->variables["gB"].array();
            this->variables["vB"] = (gamma * this->variables["cB"].array()) +
                                    ((1.0 - gamma) * (deltaParameters2.unaryExpr([](double x) {
                                        return std::pow(x, 2);
                                    }).array()));
            this->variables["b"] = this->variables["b"].array() + deltaParameters2.array();
        }
    }
}
