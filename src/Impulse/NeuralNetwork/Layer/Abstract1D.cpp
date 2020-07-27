#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract1D::Abstract1D() : Abstract() {}

            void Abstract1D::configure() {
                // initialize weights
                this->W.resize(this->height, this->width);
                this->W.setRandom();
                this->W = this->W.unaryExpr([this](const double x) {
                    return x * sqrt(2.0 / this->width);
                });

                // initialize bias
                this->b.resize(this->height);
                this->b.setRandom();
                this->b = this->b.unaryExpr([this](const double x) {
                    return x * sqrt(2.0 / this->width);
                });

                this->gW.resize(this->height, this->width);
                this->gb.resize(this->height);
            }

            bool Abstract1D::is1D() {
                return true;
            }

            bool Abstract1D::is3D() {
                return false;
            }

            void Abstract1D::transition(Layer::LayerPointer prevLayer) {
                if (prevLayer->is1D()) {
                    this->setPrevSize(prevLayer->getSize());
                } else if (prevLayer->is3D()) {
                    this->setPrevSize(prevLayer->getOutputWidth() *
                                      prevLayer->getOutputHeight() *
                                      prevLayer->getOutputDepth());
                }
            }
        }
    }
}