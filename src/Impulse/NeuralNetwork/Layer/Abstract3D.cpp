#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract3D::Abstract3D() : Abstract() {}

            bool Abstract3D::is1D() {
                return false;
            }

            bool Abstract3D::is3D() {
                return true;
            }

            void Abstract3D::transition(Layer::LayerPointer prevLayer) {
                if (prevLayer->is3D()) {
                    this->setSize(prevLayer->getOutputHeight(),
                                  prevLayer->getOutputWidth(),
                                  prevLayer->getOutputDepth());
                }
            }
        }
    }
}