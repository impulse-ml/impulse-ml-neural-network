#include "src/Impulse/Dataset/include.h"

using namespace Impulse::Dataset;
using namespace std::chrono;

char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

bool cmdOptionExists(char **begin, char **end, const std::string &option) {
    return std::find(begin, end, option) != end;
}

int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now();

    if (cmdOptionExists(argv, argv + argc, "-i")) {
        char *filenameIn = getCmdOption(argv, argv + argc, "-i");

        if (filenameIn) {
            DatasetBuilder::CSVBuilder builder(filenameIn);
            Dataset dataset = builder.build();

            if (cmdOptionExists(argv, argv + argc, "--modifier") && cmdOptionExists(argv, argv + argc, "--columns")) {
                char * modifier = getCmdOption(argv, argv + argc, "--modifier");
                char * columns = getCmdOption(argv, argv + argc, "--columns");

                if (modifier && columns) {
                    std::stringstream test(columns);
                    std::string segment;

                    while (std::getline(test, segment, ',')) {
                        if (T_String(modifier) == "category") {
                            DatasetModifier::Modifier::Category modifier2(dataset);
                            modifier2.applyToColumn(std::atoi(segment.c_str()));
                        } else if (T_String(modifier) == "minmax") {
                            DatasetModifier::Modifier::MinMaxScaling modifier2(dataset);
                            modifier2.applyToColumn(std::atoi(segment.c_str()));
                        } else if (T_String(modifier) == "zscores") {
                            DatasetModifier::Modifier::ZScoresScaling modifier2(dataset);
                            modifier2.applyToColumn(std::atoi(segment.c_str()));
                        } else if (T_String(modifier) == "missing") {
                            DatasetModifier::Modifier::MissingData modifier2(dataset);
                            modifier2.applyToColumn(std::atoi(segment.c_str()));
                        }
                    }
                }
            }

            if (cmdOptionExists(argv, argv + argc, "-o")) {
                char *filenameOut = getCmdOption(argv, argv + argc, "-o");

                if (filenameOut) {
                    DatasetExporter exporter(dataset);
                    exporter.exportToFile(filenameOut);
                }
            }

            if (cmdOptionExists(argv, argv + argc, "-v")) {
                dataset.out();
                std::cout << std::endl << "Number of rows: " << dataset.getSize() << std::endl;
                std::cout << "Number of cols: " << dataset.getColumnsSize() << std::endl << "---" << std::endl
                          << std::endl;
            }
        }
    }

    auto end = std::chrono::system_clock::now();
    long elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Done at: " << elapsed_seconds << " ms." << std::endl;

    return 0;
}