#ifndef IMPULSE_NEURALNETWORK_MATH_COMMON_H
#define IMPULSE_NEURALNETWORK_MATH_COMMON_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            typedef Eigen::MatrixXd T_Matrix;
            typedef Eigen::VectorXd T_Vector;
            typedef std::vector<double> T_RawVector;

            /**
             * Translates Eigen3 vector to std::vector for export.
             * @param vec
             * @return
             */
            Math::T_RawVector vectorToRaw(Math::T_Vector &vec);

            /**
             * Translates std::vector to Eigen3 vector.
             * @param vec
             * @return
             */
            Math::T_Vector rawToVector(Math::T_RawVector &vec);
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_MATH_COMMON_H
