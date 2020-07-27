#pragma once

#include "../../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetModifier {

            namespace Modifier {

                class Callback : public Abstract {
                protected:
                    std::function<T_String(T_String)> callback;

                public:
                    explicit Callback(Dataset &dataset) : Abstract(dataset) {}

                    void setCallback(std::function<T_String(T_String)> cb);

                    void applyToColumn(int columnIndex) override;
                };
            }
        }
    }
}
