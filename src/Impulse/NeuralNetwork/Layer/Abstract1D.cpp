#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract1D::Abstract1D() : Abstract() {}

            void Abstract1D::configure() {
                // initialize weights
                this->computation->resize("W", this->height, this->width);
                this->computation->randomInit("W", width);

                // initialize bias
                this->computation->resize("b", this->height, 1);
                this->computation->randomInit("b", width);

                // initialize gradient
                this->computation->resize("gW", this->height, this->width);
                this->computation->setZero("gW");

                this->computation->resize("gB", this->height, 1);
                this->computation->setZero("gB");

                // initialize optimizer variables
                this->computation->resize("cW", this->height, this->width);
                this->computation->setZero("cW");

                this->computation->resize("cB", this->height, 1);
                this->computation->setZero("cB");

                this->computation->resize("vW", this->height, this->width);
                this->computation->setZero("vW");

                this->computation->resize("vB", this->height, 1);
                this->computation->setZero("vB");

                this->computation->resize("wW", this->height, this->width);
                this->computation->setZero("wW");

                this->computation->resize("wB", this->height, 1);
                this->computation->setZero("wB");
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