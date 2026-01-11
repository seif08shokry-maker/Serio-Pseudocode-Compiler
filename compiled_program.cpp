#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <thread>

int main() {
    int Size = 8;
    std::vector<int> Numbers(Size, 0);
    int i = 1;
    int j = 1;
    int temp = 0;
    bool swapped = true;
    int comparisons = 0;
    int swaps = 0;
    std::cout << "Enter 8 numbers to sort:" << std::endl;
    for (int i = 1; i <= Size; ++i) {
        std::cout << "Enter number " << std::endl;
        std::cout << i << std::endl;
        std::cout << ": " << std::endl;
        std::cin >> Numbers[i];
    }
    std::cout << "Original array:" << std::endl;
    for (int i = 1; i <= Size; ++i) {
        std::cout << Numbers[i] << std::endl;
        std::cout << " " << std::endl;
    }
    std::cout << "" << std::endl;
    for (int i = 1; i <= (Size - 1); ++i) {
        swapped = false;
        for (int j = 1; j <= (Size - i); ++j) {
            comparisons = (comparisons + 1);
            if ((Numbers[j] > Numbers[(j + 1)])) {
                temp = Numbers[j];
                Numbers[j] = Numbers[(j + 1)];
                Numbers[(j + 1)] = temp;
                swapped = true;
                swaps = (swaps + 1);
            }
        }
        if ((swapped == false)) {
            std::cout << "Array became sorted after " << std::endl;
            std::cout << i << std::endl;
            std::cout << " passes" << std::endl;
            i = Size;
        }
    }
    std::cout << "Sorted array:" << std::endl;
    for (int i = 1; i <= Size; ++i) {
        std::cout << Numbers[i] << std::endl;
        std::cout << " " << std::endl;
    }
    std::cout << "" << std::endl;
    std::cout << "Total comparisons: " << std::endl;
    std::cout << comparisons << std::endl;
    std::cout << "Total swaps: " << std::endl;
    std::cout << swaps << std::endl;
    std::cout << "Sorting performance analysis:" << std::endl;
    if ((comparisons == 1)) {
        std::cout << "Excellent - minimal comparisons needed" << std::endl;
    }
    else {
        if ((comparisons == 2)) {
            std::cout << "Very good sorting efficiency" << std::endl;
        }
        else {
            if ((comparisons == 3)) {
                std::cout << "Good sorting efficiency" << std::endl;
            }
            else {
                if ((comparisons == 4)) {
                    std::cout << "Moderate sorting efficiency" << std::endl;
                }
                else {
                    std::cout << "Standard bubble sort performance" << std::endl;
                }
            }
        }
    }
    if ((swaps == 0)) {
        std::cout << "Array was already sorted!" << std::endl;
    }
    else {
        if ((swaps < 5)) {
            std::cout << "Array was nearly sorted" << std::endl;
        }
        else {
            if ((swaps > 20)) {
                std::cout << "Array required significant sorting" << std::endl;
            }
            else {
                std::cout << "Array had moderate disorder" << std::endl;
            }
        }
    }
    return 0;
}
