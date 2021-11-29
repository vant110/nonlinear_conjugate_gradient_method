#include <iostream>
#include <io.h>
#include <fcntl.h>
#include <chrono>
#include <omp.h>

int main()
{
    _setmode(_fileno(stdout), _O_U16TEXT);
    _setmode(_fileno(stdin), _O_U16TEXT);
    _setmode(_fileno(stderr), _O_U16TEXT);

    const bool isDebug = false;
    int n; // Размер матрицы.
    double* mA;
    double* vB;
    double* vX, * vXPrev;
    double* vD, * vDPrev;
    double* vG, * vGPrev;
    double* vTemp;

#pragma region Ввод значений
    std::wcout << L"Введите размер матриц: ";
    std::wcin >> n;
    {
        int nThread = 8;
        //std::wcout << L"Введите количество потоков: ";
        //std::wcin >> nThread;
        omp_set_dynamic(0);
        omp_set_num_threads(nThread);
    }
#pragma endregion
#pragma region Выделение памяти
    mA = new double[n * n];
    vB = new double[n];
    vX = new double[n];
    vD = new double[n];
    vG = new double[n];
    vXPrev = new double[n];
    vDPrev = new double[n];
    vGPrev = new double[n];
    vTemp = new double[n];
#pragma endregion
#pragma region Инициализация
    if (isDebug) std::wcout << L"Матрица А:" << std::endl;
    /*
    mA[0] = 3;
    mA[1] = -1;
    mA[2] = -1;
    mA[3] = 3;*/
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mA[i * n + j] = i + j;
            if (i + j == 0) mA[0] = 2;
            if (isDebug) std::wcout << mA[i * n + j] << '\t';
        }
        if (isDebug) std::wcout << std::endl;
    }
    if (isDebug) std::wcout << std::endl;
    if (isDebug) std::wcout << L"Вектор b:" << std::endl;
    /*
    vB[0] = 3;
    vB[1] = 7;*/
    for (int i = 0; i < n; i++) {
        vB[i] = i;
        if (i == 0) vB[0] = 1;
        vXPrev[i] = 0;
        vDPrev[i] = 0;
        vGPrev[i] = -vB[i];
        if (isDebug) std::wcout << vB[i] << '\t';
    }
    if (isDebug) std::wcout << std::endl << std::endl;
#pragma endregion

    auto beginTime = std::chrono::steady_clock::now();
#pragma region Находим решение СЛАУ
    int iter = 1, maxIter = n + 1;
    //float accuracy = 0.0001f;

    do {
        if (iter > 1) {
            std::swap(vXPrev, vX);
            std::swap(vGPrev, vG);
            std::swap(vDPrev, vD);
        }
        //compute gradient
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            vG[i] = -vB[i];
            for (int j = 0; j < n; j++)
                vG[i] += mA[i * n + j] * vXPrev[j];
        }
        //compute direction
        double IP1 = 0, IP2 = 0;
#pragma omp parallel for reduction(+:IP1,IP2)
        for (int i = 0; i < n; i++) {
            IP1 += vG[i] * vG[i];
            IP2 += vGPrev[i] * vGPrev[i];
        }
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            vD[i] = -vG[i] + vDPrev[i] * IP1 / IP2;
        }
        //compute size step
        IP1 = 0;
        IP2 = 0;
#pragma omp parallel for reduction(+:IP1,IP2)
        for (int i = 0; i < n; i++) {
            vTemp[i] = 0;
            for (int j = 0; j < n; j++) {
                vTemp[i] += mA[i * n + j] * vD[j];
            }
            IP1 += vD[i] * vG[i];
            IP2 += vD[i] * vTemp[i];
        }
        double step = -IP1 / IP2;
        //compute new approximation
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            vX[i] = vXPrev[i] + step * vD[i];
        }
        iter++;
        //} while ((Dest(vXPrev, vX, n) > accuracy) && (iter < maxIter));
    } while (iter < maxIter);
    if (isDebug) std::wcout << L"Искомый вектор x:" << std::endl;
    for (int i = 0; i < n; i++) {
        if (isDebug) std::wcout << vX[i] << '\t';
    }
    if (isDebug) std::wcout << std::endl;
#pragma endregion
    auto endTime = std::chrono::steady_clock::now();
    auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - beginTime);
    std::wcout << L"Длительность работы программы: "
        << elapsed_ns.count() / 1000000000.
        << L" сек." << std::endl;

#pragma region Освобождение памяти
    delete[] mA;
    delete[] vB;
    delete[] vX;
    delete[] vD;
    delete[] vG;
    delete[] vXPrev;
    delete[] vDPrev;
    delete[] vGPrev;
    delete[] vTemp;
#pragma endregion
}