#include <vector>
#include <iostream>

using namespace std;

using Tensor3D = vector<vector<vector<int>>>;
using Tensor2D = vector<vector<int>>;

// умножение матриц
Tensor2D multiply(const Tensor2D& a, const Tensor2D& b) {
    Tensor2D res(a.size(), vector<int>(b[0].size()));
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < b[0].size(); j++) {
            for (size_t k = 0; k < a[0].size(); k++) {
                res[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return res;
}

// input - двумерное изображение, HxWxC - размерности
// blockX, blockY - размеры фильтра
Tensor2D im2col(const Tensor3D& input, size_t blockX, size_t blockY) {
    size_t ix = input.size();
    size_t iy = input[0].size();
    size_t ic = input[0][0].size();
    size_t rx = ix + 1 - blockX;
    size_t ry = iy + 1 - blockY;
    Tensor2D result(blockX * blockY * ic, vector<int>(rx * ry));
    size_t cnt = 0;
    for (size_t x = 0; x + blockX <= input.size(); x++) {
        for (size_t y = 0; y + blockY <= input[0].size(); y++) {
            for (size_t c = 0; c < ic; c++) {
                for (size_t i = 0; i < blockX; i++) {
                    for (size_t j = 0; j < blockY; j++) {
                        result[c * blockX * blockY + i * blockY + j][cnt] = input[x + i][y + j][c];
                    }
                }
            }
            cnt++;
        }
    }
    return result;
}

Tensor3D col2im(const Tensor2D& input, size_t imgX, size_t imgY, size_t kx, size_t ky) {
    // каждая строка тензора input соответствует каналу
    // необходимо строку преобразовать в матрицу
    size_t kc = input.size();
    size_t ix = input[0].size();

    // количество строк, которое должно быть в матрице для каждого канала
    // итого на выходе функции должен получиться тензор [kc, valuesInRow, ix / valuesInRow] 
    // kc - количество каналов
    // valuesInRow, ix / valuesInRow - размер матрицы по каждому каналу 
    size_t valuesInRow = ix / (imgX - kx + 1);

    Tensor3D result(kc);
    for (size_t c = 0; c < kc; c++) {
        result[c] = Tensor2D(ix / valuesInRow, vector<int>(valuesInRow));
        for (size_t i = 0; i < ix; i++) {
            result[c][i / valuesInRow][i % valuesInRow] = input[c][i];
        }
    }
    
    return result;
}

// input - двумерное изображение, HxWxC - размерности
// kernels - фильтры, первая размерность - количество фильтров,
//                    вторая, третья - размеры фильтров
Tensor3D im2colConvLayer(const Tensor3D& input, const Tensor3D& kernels) {
    size_t kn = kernels.size();
    size_t kx = kernels[0].size();
    size_t ky = kernels[0][0].size();
    size_t ic = input[0][0].size();
    auto img = im2col(input, kx, ky);
    Tensor2D convertedKernels(kn);
    for (size_t n = 0; n < kn; n++) {
        convertedKernels[n].reserve(kx * ky * ic);
        for (size_t c = 0; c < ic; c++) {
            for (size_t x = 0; x < kx; x++) {
                for (size_t y = 0; y < ky; y++) {
                    convertedKernels[n].push_back(kernels[n][x][y]);
                }
            }
        }
    }
    auto convRes = multiply(convertedKernels, img);

    return col2im(convRes, input.size(), input[0].size(), kx, ky);
}

// референсная конволюция тензора с одним фильтром
Tensor2D referenceConvLayer(const Tensor3D& input, const Tensor2D& kernel) {
    size_t kx = kernel.size();
    size_t ky = kernel[0].size();
    Tensor2D res(input.size() - kx + 1, vector<int>(input[0].size() - ky + 1));
    for (size_t x = 0; x + kx <= input.size(); x++) {
        for (size_t y = 0; y + ky <= input[0].size(); y++) {
            for (size_t i = 0; i < kx; i++) {
                for (size_t j = 0; j < ky; j++) {
                    for (size_t c = 0; c < input[0][0].size(); c++) {
                        res[x][y] += kernel[i][j] * input[x + i][y + j][c];
                    }
                }
            }
        }
    }
    return res;
}

void printTensor3D(const Tensor3D& tensor) {
    for (size_t i = 0; i < tensor.size(); i++) {
        for (size_t j = 0; j < tensor[0].size(); j++) {
            for (size_t c; c < tensor[0][0].size(); c++) {
                cout << tensor[i][j][c] << " ";
            }
        }
        cout << "\n";
    }
}

void printTensor2D(const Tensor2D& tensor) {
    for (size_t i = 0; i < tensor.size(); i++) {
        for (size_t j = 0; j < tensor[0].size(); j++) {
            cout << tensor[i][j] << " ";
        }
        cout << "\n";
    }
}

int main() {
    // freopen("output.txt", "w", stdout);
    cout << "Case 1:\n\n";
    {
        // HxWxC : 3x4x2
        Tensor3D img = {{{1, 13}, {2, 14}, {3, 15}, {4, 16}},
                        {{5, 17}, {6, 18}, {7, 19}, {8, 20}},
                        {{9, 21}, {10, 22}, {11, 23}, {12, 24}}};
        cout << "Tensor:\n";
        printTensor3D(img);

        cout << "im2Col with 2x3 kernel:\n";
        printTensor2D(im2col(img, 2, 3));
        cout << "im2Col with 3x2 kernel:\n";
        printTensor2D(im2col(img, 3, 2));
        cout << "im2Col with 3x3 kernel:\n";
        printTensor2D(im2col(img, 3, 3));
    }

    cout << "\n\nCase 2:\n\n";
    {
        // HxWxC : 3x3x3
        Tensor3D img = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                        {{10, 11, 12}, {13, 14, 15}, {16, 17, 18}},
                        {{19, 20, 21}, {22, 23, 24}, {25, 26, 27}}};
        Tensor3D kernels = {{{1, 2},
                             {3, 4}},
                            {{4, 3},
                             {2, 1}}};
        
        cout << "Conv Layer with 2x2x2 filters:\n";
        auto res = im2colConvLayer(img, kernels);
        size_t channels = kernels.size();
        for (size_t c = 0; c < channels; c++) {
            if (res[c] != referenceConvLayer(img, kernels[c])) {
                cout << "Incorrect im2colConvLayer\n";
                return 1;
            }
            cout << "channel: " << c << "\n";
            printTensor2D(res[c]);
        }
    }

    cout << "\n\nCase 3:\n\n";
    {
        // HxWxC : 3x4x3
        Tensor3D img = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
                        {{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}},
                        {{25, 26, 27}, {28, 29, 30}, {31, 32, 33}, {34, 35, 36}}};
        Tensor3D kernels = {{{1, 2, 3},
                             {4, 5, 6}},
                            {{6, 5, 4},
                             {3, 2, 1}}};
        
        cout << "Conv Layer with 2x2x3 filters:\n";
        auto res = im2colConvLayer(img, kernels);
        size_t channels = kernels.size();
        for (size_t c = 0; c < channels; c++) {
            if (res[c] != referenceConvLayer(img, kernels[c])) {
                cout << "Incorrect im2colConvLayer\n";
                return 1;
            }
            cout << "channel: " << c << "\n";
            printTensor2D(res[c]);
        }
    }

    cout << "\n\nCase 4:\n\n";
    {
        // HxWxC : 4x4x1
        Tensor3D img = {{{1}, {2}, {3}, {4}},
                        {{5}, {6}, {7}, {8}},
                        {{9}, {10}, {11}, {12}},
                        {{13}, {14}, {15}, {16}}};
        Tensor3D kernels = {{{1, 2, 3},
                             {4, 5, 6},
                             {7, 8, 9}},
                            {{9, 8, 7},
                             {6, 5, 4},
                             {3, 2, 1}}};
        
        cout << "Conv Layer with 2x3x3 filters:\n";
        auto res = im2colConvLayer(img, kernels);
        size_t channels = kernels.size();
        for (size_t c = 0; c < channels; c++) {
            if (res[c] != referenceConvLayer(img, kernels[c])) {
                cout << "Incorrect im2colConvLayer\n";
                return 1;
            }
            cout << "channel: " << c << "\n";
            printTensor2D(res[c]);
        }
    }

    cout << "\n\nCase 5:\n\n";
    {
        // HxWxC : 3x4x3
        Tensor3D img = {{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
                        {{13, 14, 15}, {16, 17, 18}, {19, 20, 21}, {22, 23, 24}},
                        {{25, 26, 27}, {28, 29, 30}, {31, 32, 33}, {34, 35, 36}}};
        Tensor3D kernels = {{{1, 2},
                             {3, 4}},
                            {{4, 3},
                             {2, 1}},
                            {{8, 2},
                             {2, 5}}};
        
        cout << "Conv Layer with 3x2x2 filters:\n";
        auto res = im2colConvLayer(img, kernels);
        size_t channels = kernels.size();
        for (size_t c = 0; c < channels; c++) {
            if (res[c] != referenceConvLayer(img, kernels[c])) {
                cout << "Incorrect im2colConvLayer\n";
                return 1;
            }
            cout << "channel: " << c << "\n";
            printTensor2D(res[c]);
        }
    }
    return 0;
}