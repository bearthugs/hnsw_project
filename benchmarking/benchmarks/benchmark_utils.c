#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Read .fvecs format
float* load_fvecs(const char* path, int* n_out, int* dim_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fvecs file"); return NULL; }

    int dim;
    fread(&dim, 4, 1, f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = size / ((dim + 1) * 4);
    float* data = (float*)malloc(n * dim * sizeof(float));

    for (int i = 0; i < n; i++) {
        int tmp;
        fread(&tmp, 4, 1, f);
        fread(data + i * dim, 4, dim, f);
    }
    fclose(f);
    *n_out = n;
    *dim_out = dim;
    return data;
}

// Read .ivecs format (ground truth)
int* load_ivecs(const char* path, int* n_out, int* k_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("ivecs file"); return NULL; }

    int k;
    fread(&k, 4, 1, f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = size / ((k + 1) * 4);
    int* data = (int*)malloc(n * k * sizeof(int));

    for (int i = 0; i < n; i++) {
        int tmp;
        fread(&tmp, 4, 1, f);
        fread(data + i * k, 4, k, f);
    }
    fclose(f);
    *n_out = n;
    *k_out = k;
    return data;
}

double get_time_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

float compute_recall(int* result_labels, int* ground_truth, int nq, int k) {
    int correct = 0;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            int found = 0;
            for (int gt = 0; gt < k; gt++) {
                if (result_labels[i * k + j] == ground_truth[i * k + gt]) {
                    found = 1;
                    break;
                }
            }
            correct += found;
        }
    }
    return (float)correct / (nq * k);
}
