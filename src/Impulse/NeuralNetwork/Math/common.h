#pragma once

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            typedef std::vector<double> T_RawVector;

            /**
             * Translates Eigen3 vector to std::vector for export.
             * @param vec
             * @return
             */
            Math::T_RawVector vectorToRaw(Eigen::VectorXd &vec);

            /**
             * Translates std::vector to Eigen3 vector.
             * @param vec
             * @return
             */
            Eigen::VectorXd rawToVector(Math::T_RawVector &vec);
        }
    }
}
