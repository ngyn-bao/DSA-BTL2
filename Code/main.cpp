#include <iostream>
#include <iomanip>
#include <sstream> //lệnh compiler: g++ -Idemo -Iinclude main.cpp
#include <string>  //lệnh run: ./main.exe hoặc ./a.exe (tùy file thực thi máy xuất ra sau khi compile thành công)
#include <fstream>
#include "include/list/listheader.h"
#include "demo/hash/xMapDemo.h"
#include "demo/heap/HeapDemo.h"
#include "include/hash/xMap.h"

using namespace std;

int main(int argc, char **argv)
{
    ofstream outFile1("outputhash.txt");
    ofstream outFile2("outputheap.txt");

    if (!outFile1)
    {
        cerr << "Không thể mở file outputhash.txt để ghi!" << endl;
        return 1;
    }

    if (!outFile2)
    {
        cerr << "Không thể mở file outputheap.txt để ghi!" << endl;
        return 1;
    }

    // TEST HASH :
    void (*hashDemos[])() = {0, hashDemo1, hashDemo2, hashDemo3, hashDemo4, hashDemo5, hashDemo6};
    outFile1 << "TEST HASH DEMO:......................................." << endl;

    for (int i = 1; i <= 6; i++)
    { // test hash i<=7
        outFile1 << "Demo " << i << "-------------------------" << endl;
        // outFile1 << "Demo " << 3 << "-------------------------" << endl;
        outFile1 << endl;

        streambuf *coutBuffer = cout.rdbuf();
        cout.rdbuf(outFile1.rdbuf());
        hashDemos[i]();
        // hashDemo3();
        cout.rdbuf(coutBuffer);
    }

    outFile1.close();
    // Test heap:
    void (*heapDemos[])() = {0, heapDemo1, heapDemo2, heapDemo3};
    outFile2 << "TEST HEAP DEMO:......................................." << endl;

    for (int i = 1; i <= 3; i++)
    { // test heap i<=3
        outFile2 << "Demo " << i << "-------------------------" << endl;
        outFile2 << endl;

        streambuf *coutBuffer = cout.rdbuf();
        cout.rdbuf(outFile2.rdbuf());
        heapDemos[i]();
        cout.rdbuf(coutBuffer);
    }

    outFile2.close();

    return 0;
}
