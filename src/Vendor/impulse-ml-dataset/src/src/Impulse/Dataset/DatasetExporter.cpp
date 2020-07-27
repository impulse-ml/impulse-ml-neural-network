#include "include.h"

namespace Impulse {

    namespace Dataset {

        DatasetExporter::DatasetExporter(Dataset &dataset) : dataset(dataset) {
        }

        void DatasetExporter::exportToFile(T_String fileName) {
            std::ofstream file;
            file.open(fileName);

            for (int i = 0; i < this->dataset.getSize(); i += 1) {
                for (int j = 0; j < this->dataset.getColumnsSize(); j += 1) {
                    file << this->dataset.getSampleAt(i)->getColumnToString(j);
                    if (j < this->dataset.getColumnsSize() - 1) {
                        file << ",";
                    }
                }
                file << "\n";
            }

            file.close();
        }
    }
}
