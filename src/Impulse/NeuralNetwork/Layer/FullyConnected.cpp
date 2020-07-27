#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            FullyConnected::FullyConnected() : Conv() {};

            const T_String FullyConnected::getType() {
                return TYPE_FULLYCONNECTED;
            }

            void FullyConnected::transition(Layer::LayerPointer prevLayer) {
                if (prevLayer->is3D()) {
                    if (prevLayer->getType() == Layer::TYPE_MAXPOOL) {
                        auto layer = (Layer::MaxPool *) prevLayer.get();

                        this->filterSize = layer->getOutputWidth();
                        this->padding = 0;
                        this->stride = 1;
                        this->width = layer->getOutputWidth();
                        this->height = layer->getOutputHeight();
                        this->depth = layer->getOutputDepth();
                        this->numFilters = layer->getOutputWidth() * layer->getOutputHeight() * layer->getOutputDepth();
                    }
                }
            }

            void FullyConnected::setSize(T_Size value) {
                static_assert("No setSize for FULLYCONNECTED layer.", "");
            }

            void FullyConnected::setFilterSize(T_Size value) {
                static_assert("No setFilterSize for FULLYCONNECTED layer.", "");
            }

            void FullyConnected::setPadding(T_Size value) {
                static_assert("No setPadding for FULLYCONNECTED layer.", "");
            }

            void FullyConnected::setStride(T_Size value) {
                static_assert("No setStride for FULLYCONNNECTED layer.", "");
            }

            void FullyConnected::setWidth(T_Size value) {
                static_assert("No setWidth for FULLYCONNNECTED layer.", "");
            }

            void FullyConnected::setHeight(T_Size value) {
                static_assert("No setHeight for FULLYCONNNECTED layer.", "");
            }

            void FullyConnected::setDepth(T_Size value) {
                static_assert("No setDepth for FULLYCONNNECTED layer.", "");
            }

            void FullyConnected::setNumFilters(T_Size value) {
                static_assert("No setNumFilters for FULLYCONNECTED layer.", "");
            }
        }
    }
}
