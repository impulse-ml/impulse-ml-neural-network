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

            if (cmdOptionExists(argv, argv + argc, "--input-columns")) {
                char *inputColumns = getCmdOption(argv, argv + argc, "--input-columns");
                char *outputColumns = getCmdOption(argv, argv + argc, "--output-columns");

                // slice dataset to input and output set
                DatasetModifier::DatasetSlicer datasetSlicer(dataset);

                if (inputColumns) {
                    std::stringstream test(inputColumns);
                    std::string segment;

                    while (std::getline(test, segment, ',')) {
                        datasetSlicer.addInputColumn(std::atoi(segment.c_str()));
                    }
                }
                if (outputColumns) {
                    std::stringstream test(outputColumns);
                    std::string segment;

                    while (std::getline(test, segment, ',')) {
                        datasetSlicer.addOutputColumn(std::atoi(segment.c_str()));
                    }
                }

                SlicedDataset _dataset = datasetSlicer.slice();

                if (cmdOptionExists(argv, argv + argc, "-o")) {
                    char *filenameOut = getCmdOption(argv, argv + argc, "-o");
                    T_String filename = T_String(filenameIn).substr(T_String(filenameIn).find_last_of("/") + 1);

                    if (filenameOut) {
                        DatasetExporter exporter(_dataset.input);
                        exporter.exportToFile(T_String(filenameOut).append("/").append(filename).append(".INPUT.csv"));

                        DatasetExporter exporter2(_dataset.output);
                        exporter2.exportToFile(T_String(filenameOut).append("/").append(filename).append(".OUTPUT.csv"));
                    }

                    if (cmdOptionExists(argv, argv + argc, "-v")) {
                        std::cout << "INPUT: " << std::endl;
                        std::cout << std::endl << "Number of rows: " << _dataset.input.getSize() << std::endl;
                        std::cout << "Number of cols: " << _dataset.input.getColumnsSize() << std::endl << "---"
                                  << std::endl;
                        _dataset.input.out();
                        std::cout << "OUTPUT: " << std::endl;
                        std::cout << std::endl << "Number of rows: " << _dataset.output.getSize() << std::endl;
                        std::cout << "Number of cols: " << _dataset.output.getColumnsSize() << std::endl << "---"
                                  << std::endl;
                        _dataset.output.out();
                    }
                }
            }
        }
    }

    auto end = std::chrono::system_clock::now();
    long elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Done at: " << elapsed_seconds << " ms." << std::endl;

    return 0;
}