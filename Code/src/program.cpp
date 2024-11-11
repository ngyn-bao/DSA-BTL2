#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
using namespace std;

#include <filesystem> //require C++17
namespace fs = std::filesystem;

#include "list/listheader.h"
#include "sformat/fmt_lib.h"
#include "tensor/xtensor_lib.h"
#include "ann/annheader.h"
#include "loader/dataset.h"
#include "loader/dataloader.h"
#include "config/Config.h"
#include "dataset/DSFactory.h"
#include "optim/Adagrad.h"
#include "optim/Adam.h"
#include "modelzoo/twoclasses.h"
#include "modelzoo/threeclasses.h"
#include "modelzoo/mlpDemo.h"

int main(int argc, char **argv)
{
    // dataloader:
    // case_data_wo_label_1();
    // case_data_wi_label_1();
    // case_batch_larger_nsamples();

    // Classification:
    cout << "Two classes \n";
    twoclasses_classification();
    cout << "Three classes \n";
    threeclasses_classification();
    int num;
    cin >> num;

    switch (num)
    {
    case 1:
        mlpDemo1();
        break;
    case 2:
        mlpDemo2();
        break;
    case 3:
        mlpDemo3();
        break;
    default:
        break;
    }

    return 0;
}
