#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <vector>
#include <iostream>
#include <string>
#include <functional>
#include <fstream>
#include <algorithm>
#include <limits>
#include <cmath>
#include <map>
#include <cstdarg>
#include <chrono>
#include <utility>
#include <memory>

#include "common.h"
#include "DatasetSample.h"
#include "Dataset.h"
#include "DatasetBuilder/Abstract.h"
#include "DatasetBuilder/CSVBuilder.h"
#include "DatasetModifier/Modifier/Abstract.h"
#include "DatasetModifier/DatasetSlicer.h"
#include "DatasetModifier/DatasetSplitter.h"
#include "DatasetModifier/Modifier/Callback.h"
#include "DatasetModifier/Modifier/Category.h"
#include "DatasetModifier/Modifier/CategoryId.h"
#include "DatasetModifier/Modifier/MinMaxScaling.h"
#include "DatasetModifier/Modifier/MissingData.h"
#include "DatasetModifier/Modifier/ZScoresScaling.h"
#include "DatasetExporter.h"
