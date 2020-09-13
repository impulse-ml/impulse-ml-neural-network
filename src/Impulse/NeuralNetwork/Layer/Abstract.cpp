#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract::Abstract() {
                this->computation = new Computation();

                this->computation->initialize("W");
                this->computation->initialize("b");
                this->computation->initialize("A");
                this->computation->initialize("Z");
                this->computation->initialize("gW");
                this->computation->initialize("gB");
                this->computation->initialize("cW");
                this->computation->initialize("cB");
                this->computation->initialize("vW");
                this->computation->initialize("vB");
                this->computation->initialize("wW");
                this->computation->initialize("wB");
            };

            Abstract::~Abstract() {
                delete this->computation;
            }

            Eigen::MatrixXd Abstract::forward(const Eigen::MatrixXd &input) {
                this->computation->forward(input);
                this->activation();
                return this->computation->getVariable("A");
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

            double Abstract::penalty() {
                return this->computation->penalty();
            }

            Computation *Abstract::getComputation() {
                return this->computation;
            }
        }
    }
}
