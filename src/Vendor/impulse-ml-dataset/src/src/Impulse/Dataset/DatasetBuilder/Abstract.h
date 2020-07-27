#pragma once

#include "../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetBuilder {

            class Abstract {
            protected:
                DatasetSampleContainer createSample(T_StringVector vec);

            public:
                virtual Dataset build() = 0;

                virtual ~Abstract() = default;
            };
        }
    }
}
