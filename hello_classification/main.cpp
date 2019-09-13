// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "iostream"
#include <iomanip>
#include <vector>
#include <memory>
#include <string>
#include <cstdlib>
#include "fstream"
#include "reid.h"


#ifndef UNICODE
#define tcout std::cout
#define _T(STR) STR
#else
#define tcout std::wcout
#endif

#ifndef UNICODE
int main(int argc, char *argv[]) {
#else
int wmain(int argc, wchar_t *argv[]) {
#endif
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 3) {
            tcout << _T("Usage : ./hello_classification <path_to_model> <test_file>") << std::endl;
            return EXIT_FAILURE;
        }
        std::ofstream output_file("rs_temp");
        auto reid = new ReID;


        std::vector<cv::Mat> images;
        std::vector<std::vector<float>> classify_results;
        cv::Mat test = cv::imread(argv[2]);
        images.push_back(test);
        images.push_back(test);
        images.push_back(test);
        images.push_back(test);

        reid->init(argv[1], 0, 0, false);

        reid->set_model_type(1);
        reid->classify(images, classify_results);

        for(int i = 0; i < classify_results.size(); ++i) {
            for (auto &f : classify_results[i]) {
                output_file << f << " ";
            }
            output_file << std::endl;
        }
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
