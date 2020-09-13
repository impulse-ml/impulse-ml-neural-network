#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            Abstract::Abstract(T_Dimension dim) {
                this->dimension = dim;
            }

            void Abstract::addLayer(Layer::LayerPointer layer) {
                this->size++;
                this->layers.push_back(layer);
            }

            Eigen::MatrixXd Abstract::forward(const Eigen::MatrixXd &input) {
                Eigen::MatrixXd output = input;

                for (auto &layer : this->layers) {
                    output = layer->forward(output);
                }

                return output;
            }

            void
            Abstract::backward(Eigen::MatrixXd &X, Eigen::MatrixXd &Y, Eigen::MatrixXd &predictions,
                               double regularization) {
                long m = X.cols();

                Eigen::MatrixXd delta = predictions.array() - Y.array();

                for (long i = this->layers.size() - 1; i >= 0; i--) {
                    auto layer = this->layers.at(static_cast<unsigned long>(i));
                    delta = layer->backpropagation->propagate(X, (T_Size) m, regularization, delta);
                }
            }

            T_Dimension Abstract::getDimension() {
                return this->dimension;
            }

            T_Size Abstract::getSize() {
                return this->size;
            }

            Layer::LayerPointer Abstract::getLayer(T_Size key) {
                return this->layers.at(key);
            }

            Eigen::VectorXd Abstract::getRolledTheta() {
                Math::T_RawVector tmp;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->getLayer(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    tmp.reserve(
                            (unsigned long) (layer->getComputation()->getVariable("W").cols() *
                                             layer->getComputation()->getVariable("W").rows()) +
                            (layer->getComputation()->getVariable("b").cols() *
                             layer->getComputation()->getVariable("b").rows()));

                    for (T_Size j = 0; j < layer->getComputation()->getVariable("W").rows(); j++) {
                        for (T_Size k = 0; k < layer->getComputation()->getVariable("W").cols(); k++) {
                            tmp.push_back(layer->getComputation()->getVariable("W")(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->getComputation()->getVariable("b").rows(); j++) {
                        for (T_Size k = 0; k < layer->getComputation()->getVariable("b").cols(); k++) {
                            tmp.push_back(layer->getComputation()->getVariable("b")(j, k));
                        }
                    }
                }

                Eigen::VectorXd result = Math::rawToVector(tmp);
                return result;
            }

            Eigen::VectorXd Abstract::getRolledGradient() {
                Math::T_RawVector tmp;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->layers.at(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    for (T_Size j = 0; j < layer->getComputation()->getVariable("gW").rows(); j++) {
                        for (T_Size k = 0; k < layer->getComputation()->getVariable("gW").cols(); k++) {
                            tmp.push_back(layer->getComputation()->getVariable("gW")(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->getComputation()->getVariable("gB").rows(); j++) {
                        for (T_Size k = 0; k < layer->getComputation()->getVariable("gB").cols(); k++) {
                            tmp.push_back(layer->getComputation()->getVariable("gB")(j, k));
                        }
                    }
                }

                Eigen::VectorXd result = Math::rawToVector(tmp);
                return result;
            }

            void Abstract::setRolledTheta(Eigen::VectorXd &theta) {
                unsigned long t = 0;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->layers.at(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    for (T_Size j = 0; j < layer->getComputation()->getVariable("W").rows(); j++) {
                        for (T_Size k = 0; k < layer->getComputation()->getVariable("W").cols(); k++) {
                            layer->getComputation()->getVariable("W")(j, k) = theta(t++);
                        }
                    }

                    for (T_Size j = 0; j < layer->getComputation()->getVariable("b").rows(); j++) {
                        for (T_Size k = 0; k < layer->getComputation()->getVariable("b").cols(); k++) {
                            layer->getComputation()->getVariable("b")(j, k) = theta(t++);
                        }
                    }
                }
            }

            double Abstract::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                return this->layers.at(this->getSize() - 1)->loss(output, predictions);
            }

            double Abstract::error(T_Size m) {
                return this->layers.at(this->getSize() - 1)->error(m);
            }

            void Abstract::debug() {

            }
        }
    }
}
