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

            Math::T_Matrix Abstract::forward(const Math::T_Matrix &input) {
                Math::T_Matrix output = input;

                for (auto &layer : this->layers) {
                    output = layer->forward(output);
                }

                return output;
            }

            void
            Abstract::backward(Math::T_Matrix X, Math::T_Matrix Y, Math::T_Matrix predictions, double regularization) {
                long m = X.cols();

                Math::T_Matrix delta = predictions.array() - Y.array();

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

            Math::T_Vector Abstract::getRolledTheta() {
                Math::T_RawVector tmp;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->getLayer(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    tmp.reserve((unsigned long) (layer->W.cols() * layer->W.rows()) + (layer->b.cols() * layer->b.rows()));

                    for (T_Size j = 0; j < layer->W.rows(); j++) {
                        for (T_Size k = 0; k < layer->W.cols(); k++) {
                            tmp.push_back(layer->W(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->b.rows(); j++) {
                        for (T_Size k = 0; k < layer->b.cols(); k++) {
                            tmp.push_back(layer->b(j, k));
                        }
                    }
                }

                Math::T_Vector result = Math::rawToVector(tmp);
                return result;
            }

            Math::T_Vector Abstract::getRolledGradient() {
                Math::T_RawVector tmp;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->layers.at(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    for (T_Size j = 0; j < layer->gW.rows(); j++) {
                        for (T_Size k = 0; k < layer->gW.cols(); k++) {
                            tmp.push_back(layer->gW(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->gb.rows(); j++) {
                        for (T_Size k = 0; k < layer->gb.cols(); k++) {
                            tmp.push_back(layer->gb(j, k));
                        }
                    }
                }

                Math::T_Vector result = Math::rawToVector(tmp);
                return result;
            }

            void Abstract::setRolledTheta(Math::T_Vector theta) {
                unsigned long t = 0;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->layers.at(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    for (T_Size j = 0; j < layer->W.rows(); j++) {
                        for (T_Size k = 0; k < layer->W.cols(); k++) {
                            layer->W(j, k) = theta(t++);
                        }
                    }

                    for (T_Size j = 0; j < layer->b.rows(); j++) {
                        for (T_Size k = 0; k < layer->b.cols(); k++) {
                            layer->b(j, k) = theta(t++);
                        }
                    }
                }
            }

            double Abstract::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                return this->layers.at(this->getSize() - 1)->loss(std::move(output), std::move(predictions));
            }

            double Abstract::error(T_Size m) {
                return this->layers.at(this->getSize() - 1)->error(m);
            }

            void Abstract::debug() {

            }
        }
    }
}
