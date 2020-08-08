#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract1D::Abstract1D() : Abstract() {}

            void Abstract1D::configure() {
                // initialize weights
                this->W.resize(this->height, this->width);
                this->W = Computation::factory().randomInit(this->W, this->width);

                // initialize bias
                this->b.resize(this->height);
                this->b = Computation::factory().randomInit(this->b, this->width);

                this->gW.resize(this->height, this->width);
                this->gW.setZero();

                this->gB.resize(this->height);
                this->gB.setZero();

                this->cW.resize(this->height, this->width);
                this->cW.setZero();

                this->cB.resize(this->height);
                this->cB.setZero();

                this->vW.resize(this->height, this->width);
                this->vW.setZero();

                this->vB.resize(this->height);
                this->vB.setZero();

                this->wW.resize(this->height, this->width);
                this->wW.setZero();

                this->wB.resize(this->height);
                this->wB.setZero();
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