#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            MaxPool::MaxPool() : Abstract3D() {}

            void MaxPool::configure() {
                // empty configure - no parameters.
            }

            void MaxPool::setFilterSize(T_Size value) {
                this->filterSize = value;
            }

            T_Size MaxPool::getFilterSize() {
                return this->filterSize;
            }

            void MaxPool::setStride(T_Size value) {
                this->stride = value;
            }

            T_Size MaxPool::getStride() {
                return this->stride;
            }

            Eigen::MatrixXd MaxPool::forward(const Eigen::MatrixXd &input) {
                this->computation->setVariable("Z", input);

                Eigen::MatrixXd result(this->getOutputWidth() * this->getOutputHeight() * this->getOutputDepth(),
                                       input.cols());

#pragma omp parallel
                for (T_Size i = 0; i < input.cols(); i++) {
                    result.col(i) = Utils::maxpool(input.col(i), this->depth,
                                                   this->height, this->width,
                                                   this->filterSize, this->filterSize,
                                                   this->stride, this->stride);
                }

                return result;
            }

            Eigen::MatrixXd MaxPool::activation() {
                static_assert("No activation for MAXPOOL layer.", "");
                return Eigen::MatrixXd(); // no activation for maxpool layer
            }

            Eigen::MatrixXd MaxPool::derivative(Eigen::MatrixXd &) {
                static_assert("No derivative for MAXPOOL layer.", "");
                return Eigen::MatrixXd(); // no derivative for maxpool layer
            }

            const T_String MaxPool::getType() {
                return TYPE_MAXPOOL;
            }

            double MaxPool::loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) {
                static_assert("No loss for MAXPOOL layer.", "");
                return 0.0;
            }

            double MaxPool::error(T_Size m) {
                static_assert("No error for MAXPOOL layer.", "");
                return 0.0;
            }

            T_Size MaxPool::getOutputHeight() {
                return (this->height - this->filterSize) / this->stride + 1;
            }

            T_Size MaxPool::getOutputWidth() {
                return (this->width - this->filterSize) / this->stride + 1;
            }

            T_Size MaxPool::getOutputDepth() {
                return this->depth;
            }
        }
    }
}
