#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../../Vendor/json/src/json.hpp"
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <chrono>
#include <functional>
#include <cmath>
#include <fstream>
#include <utility>

#include "../../Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/include.h"

#include "common.h"
#include "Math/common.h"
#include "utils.h"
#include "Trainer/common.h"
#include "AbstractComputation.h"
#include "Layer/BackPropagation/Abstract.h"
#include "Layer/BackPropagation/BackPropagation1DTo1D.h"
#include "Layer/BackPropagation/BackPropagationToMaxPool.h"
#include "Layer/BackPropagation/BackPropagationToConv.h"
#include "Layer/BackPropagation/BackPropagation3DTo1D.h"
#include "Layer/BackPropagation/Factory.h"
#include "Layer/Abstract.h"
#include "Layer/Abstract1D.h"
#include "Layer/Abstract3D.h"
#include "Network/Abstract.h"
#include "Network/ConvNetwork.h"
#include "Network/ClassifierNetwork.h"
#include "Builder/Abstract.h"
#include "Builder/ClassifierBuilder.h"
#include "Builder/ConvBuilder.h"
#include "Serializer.h"
#include "Layer/Tanh.h"
#include "Layer/Softmax.h"
#include "Layer/Relu.h"
#include "Layer/Logistic.h"
#include "Layer/Purelin.h"
#include "Layer/Conv.h"
#include "Layer/MaxPool.h"
#include "Layer/FullyConnected.h"
#include "Layer/Softplus.h"
#include "Math/Fmincg.h"
#include "ComputationCpu.h"
#include "Computation.h"
#include "Trainer/Optimizer/Abstract.h"
#include "Trainer/Optimizer/Adam.h"
#include "Trainer/Optimizer/Adagrad.h"
#include "Trainer/Optimizer/GradientDescent.h"
#include "Trainer/Optimizer/Momentum.h"
#include "Trainer/Optimizer/Nesterov.h"
#include "Trainer/Optimizer/Rmsprop.h"
#include "Trainer/Abstract.h"
#include "Trainer/ConjugateGradient.h"
#include "Trainer/Stochastic.h"
#include "Trainer/MiniBatch.h"
