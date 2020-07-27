#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract::Abstract() = default;

            Math::T_Matrix Abstract::forward(const Math::T_Matrix &input) {
                this->Z = (this->W * input).colwise() + this->b;
                this->A = this->activation(this->Z);
                return this->A;
            }

            void Abstract::setSize(T_Size value) {
                this->setHeight(value);
            }

            void Abstract::setSize(T_Size width, T_Size height, T_Size depth) {
                this->setWidth(width);
                this->setHeight(height);
                this->setDepth(depth);
            }

            void Abstract::setPrevSize(T_Size value) {
                this->setWidth(value);
            }

            void Abstract::setWidth(T_Size value) {
                this->width = value;
            }

            T_Size Abstract::getWidth() {
                return this->width;
            }

            void Abstract::setHeight(T_Size value) {
                this->height = value;
            }

            T_Size Abstract::getHeight() {
                return this->height;
            }

            void Abstract::setDepth(T_Size value) {
                this->depth = value;
            }

            T_Size Abstract::getDepth() {
                return this->depth;
            }

            T_Size Abstract::getSize() {
                return this->height;
            }

            T_Size Abstract::getOutputWidth() {
                return this->width;
            }

            T_Size Abstract::getOutputHeight() {
                return this->height;
            }

            T_Size Abstract::getOutputDepth() {
                return 1;
            }
        }
    }
}
