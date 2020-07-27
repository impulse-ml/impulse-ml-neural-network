#include "../include.h"

namespace Impulse {

    namespace Dataset {

        namespace DatasetBuilder {

            DatasetSampleContainer Abstract::createSample(T_StringVector vec) {
                return std::make_shared<Impulse::Dataset::DatasetSample>(std::move(vec));
            }
        }
    }
}
