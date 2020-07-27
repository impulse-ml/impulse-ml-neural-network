#ifndef IMPULSE_NEURALNETWORK_COMMON_H
#define IMPULSE_NEURALNETWORK_COMMON_H

#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        typedef unsigned int T_Size;
        typedef std::string T_String;

        typedef struct {
            T_Size width = 0;
            T_Size height = 0;
            T_Size depth = 0;
        } T_Dimension;
    }
}

#endif //IMPULSE_NEURALNETWORK_
