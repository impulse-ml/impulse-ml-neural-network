#pragma once

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            // fwd declarations
            class Abstract;

            /**
             * Layer pointer.
             */
            typedef std::shared_ptr<Abstract> LayerPointer;

            class Abstract {
            protected:
                T_Size width;                                                           // number of prev layer size (input)
                T_Size height;                                                          // number of neurons
                T_Size depth;                                                           // 3D YEAH
                Layer::LayerPointer previousLayer = nullptr;                            // pointer to the previous layer in the network

            public:
                Eigen::MatrixXd W;                                                       // weights
                Eigen::VectorXd b;                                                       // bias
                Eigen::MatrixXd A;                                                       // output of the layer after activation
                Eigen::MatrixXd Z;                                                       // output of the layer before activation
                Eigen::MatrixXd gW;                                                      // gradient for weights
                Eigen::VectorXd gb;                                                      // gradient for biases
                BackPropagation::BackPropagationPointer backpropagation = nullptr;      // pointer to the backpropagation algorithm
                /**
                 * Pure constructor
                 */
                Abstract();

                /**
                 * Forward propagation.
                 * @param input
                 * @return
                 */
                virtual Eigen::MatrixXd forward(const Eigen::MatrixXd &input);

                /**
                 * Calculates activated values.
                 * @param input
                 * @return
                 */
                virtual Eigen::MatrixXd activation(Eigen::MatrixXd &m) = 0;

                /**
                 * Calculates derivative. It depends on activation function.
                 * @return
                 */
                virtual Eigen::MatrixXd derivative() = 0;

                /**
                 * Getter for layer type.
                 * @return
                 */
                virtual const T_String getType() = 0;

                /**
                 * Setter for size.
                 * @param value
                 */
                virtual void setSize(T_Size value);

                /**
                 *
                 * @param width
                 * @param height
                 * @param depth
                 */
                void setSize(T_Size width, T_Size height, T_Size depth);

                /**
                 *
                 * @param value
                 */
                void setPrevSize(T_Size value);

                /**
                 *
                 * @param value
                 */
                virtual void setWidth(T_Size value);

                /**
                 *
                 * @return
                 */
                T_Size getWidth();

                /**
                 *
                 * @param value
                 */
                virtual void setHeight(T_Size value);

                /**
                 *
                 * @return
                 */
                T_Size getHeight();

                /**
                 *
                 * @param value
                 */
                virtual void setDepth(T_Size value);

                /**
                 *
                 * @return
                 */
                T_Size getDepth();

                /**
                 * Get output Rows
                 */
                virtual T_Size getOutputHeight();

                /**
                 * Get output Cols
                 */
                virtual T_Size getOutputWidth();

                /**
                 * Get depth
                 */
                virtual T_Size getOutputDepth();

                /**
                 * Getter for layer size.
                 * @return
                 */
                T_Size getSize();

                /**
                 * Loss for the last, classifier layer.
                 * @param output
                 * @param predictions
                 * @return
                 */
                virtual double loss(Eigen::MatrixXd &output, Eigen::MatrixXd &predictions) = 0;

                /**
                 * Error term for network.
                 * @param m
                 * @return
                 */
                virtual double error(T_Size m) = 0;

                /**
                 * Finish configuration of the layer
                 */
                virtual void configure() = 0;

                /**
                 * Is 2d layer
                 * @return
                 */
                virtual bool is1D() = 0;

                /**
                 * Is 3d layer
                 * @return
                 */
                virtual bool is3D() = 0;

                /**
                 * Transition
                 */
                virtual void transition(Layer::LayerPointer prevLayer) = 0;

                /**
                 * Debug.
                 */
                virtual void debug() {};

                double penalty();
            };
        }
    }
}
