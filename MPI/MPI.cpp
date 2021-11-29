#include <iostream>
#include "mpi.h"

const bool isDebug = false;
const int n = 10; // Размер квадратных матриц.

double** malloc_2d(int, int);
void free_2d(double**);
double calcElem(double**, double**, int, int);

int main(int argc, char** argv)
{ 
    int currProcess, nProcess;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &currProcess);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcess);
    const int nRowNewA = n / nProcess;
    if (currProcess == 0) {
        std::cout << "Num processes: " << nProcess << std::endl;
        // Выделение памяти.
        double** mA = malloc_2d(n, n);
        double** mB = malloc_2d(n, n);
        double** mC = malloc_2d(n, n);
        // Инициализация матриц-множителей.
        if (isDebug) std::cout << "Matrix-multiplier:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mA[i][j] = (double)i + j;
                mB[i][j] = (double)i + j;
                if (isDebug) std::cout << mA[i][j] << '\t';
            }
            if (isDebug) std::cout << std::endl;
        }
        // Отправка матриц-множителей другим процессам.
        double startTime = MPI_Wtime();
        int startRowNewA;
        for (int toProcess = 1; toProcess < nProcess; toProcess++) {
            startRowNewA = (toProcess - 1) * nRowNewA;
            MPI_Send(&(mA[startRowNewA][0]), nRowNewA * n, MPI_DOUBLE, toProcess, 0, MPI_COMM_WORLD);
            MPI_Send(&(mB[0][0]), n * n, MPI_DOUBLE, toProcess, 1, MPI_COMM_WORLD);
        }
        // Вычисление полосы результирующей матрицы.
        startRowNewA = (nProcess - 1) * nRowNewA;
        for (int i = startRowNewA; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mC[i][j] = calcElem(mA, mB, i, j);
            }
        }
        // Освобождение памяти.
        free_2d(mA);
        free_2d(mB);
        // Получение полос результирующей матрицы от других процессов.
        if (true) {
            // Получение сообщений по готовности.
            double** buffC = malloc_2d(nRowNewA, n);
            for (int i = 1; i < nProcess; i++) {
                MPI_Recv(&(buffC[0][0]), nRowNewA * n, MPI_DOUBLE, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
                startRowNewA = (status.MPI_SOURCE - 1) * nRowNewA;
                for (int j = 0; j < nRowNewA * n; j++) {
                    mC[startRowNewA + j / n][j % n] = buffC[j / n][j % n];
                }
            }
            free_2d(buffC);
        }
        else {
            // Получение сообщений по порядку.
            for (int fromProcess = 1; fromProcess < nProcess; fromProcess++) {
                startRowNewA = (fromProcess - 1) * nRowNewA;
                MPI_Recv(&(mC[startRowNewA][0]), nRowNewA * n, MPI_DOUBLE, fromProcess, 2, MPI_COMM_WORLD, &status);
            }
        }
        double endTime = MPI_Wtime();
        // Вывод результатов.
        if (isDebug) {
            std::cout << "The result of multiplying two identical matrices:" << std::endl;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    std::cout << mC[i][j] << '\t';
                }
                std::cout << std::endl;
            }
        }
        std::cout << "Duration of the program: "
            << endTime - startTime
            << " sec." << std::endl;
        // Освобождение памяти.
        free_2d(mC);
    }
    else {
        // Выделение памяти.
        double** newA = malloc_2d(nRowNewA, n);
        double** newB = malloc_2d(n, n);
        double** newC = malloc_2d(nRowNewA, n);
        // Получение матриц-множителей от главного процесса.
        MPI_Recv(&(newA[0][0]), nRowNewA * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&(newB[0][0]), n * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
        // Вычисление полосы результирующей матрицы.      
        for (int i = 0; i < nRowNewA; i++) {
            for (int j = 0; j < n; j++) {
                newC[i][j] = calcElem(newA, newB, i, j);
            }
        }
        // Отправление полосы результирующей матрицы главному процессу.  
        MPI_Send(&(newC[0][0]), nRowNewA * n, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
        // Освобождение памяти.
        free_2d(newA);
        free_2d(newB);
        free_2d(newC);
    }
    MPI_Finalize();
}

double** malloc_2d(int nRow, int nColumn) {
    double* data = (double*)malloc(nRow * nColumn * sizeof(double));
    double** arr = (double**)malloc(nRow * sizeof(double*));
    for (int i = 0; i < nRow; i++) {
        arr[i] = &(data[nColumn * i]);
    }
    return arr;
}

void free_2d(double** arr) {
    free(arr[0]);
    free(arr);
}

double calcElem(double** mA, double** mB, int i, int j) {
    double sum = 0;
    for (int k = 0; k < n; k++) {
        sum += mA[i][k] * mB[k][j];
    }
    return sum;
}