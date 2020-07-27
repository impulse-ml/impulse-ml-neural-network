#pragma once

#include "../include.h"

using namespace Impulse::Dataset;

namespace Impulse {

    namespace Dataset {

        namespace DatasetBuilder {

            enum class CSVState {
                UnquotedField, QuotedField, QuotedQuote
            };

            class CSVBuilder : public Abstract {

            protected:
                T_String path;
                std::ifstream fileHandle;

                void openFile();

                void closeFile();

                T_StringVector parseLine(T_String &line);

            public:
                explicit CSVBuilder(T_String p);

                ~CSVBuilder() override;

                Dataset build() override;
            };
        }
    }
}
