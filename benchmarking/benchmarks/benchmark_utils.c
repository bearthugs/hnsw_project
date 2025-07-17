#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Load .fvecs file (SIFT1M)
// Returns: float pointer to all vectors (malloc'ed, caller frees)
// Outputs *out_num_vectors and *out_dim
float* load_fvecs(const char* filename, int* out_num_vectors, int* out_dim) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        perror("Failed to open file");
        return NULL;
    }

    // Read dimension of first vector
    int dim = 0;
    if (fread(&dim, sizeof(int), 1, f) != 1) {
        perror("Failed to read dimension");
        fclose(f);
        return NULL;
    }
    if (dim <= 0 || dim > 10000) {
        fprintf(stderr, "Unreasonable dimension: %d\n", dim);
        fclose(f);
        return NULL;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Each vector = 4 bytes (dim) + dim * 4 bytes (floats)
    long vec_size = sizeof(int) + dim * sizeof(float);
    int num_vectors = (int)(file_size / vec_size);

    // Allocate memory for all vectors (just floats, no dimension)
    float* data = (float*)malloc(num_vectors * dim * sizeof(float));
    if (!data) {
        perror("malloc failed");
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < num_vectors; i++) {
        int d = 0;
        if (fread(&d, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Failed to read dim for vector %d\n", i);
            free(data);
            fclose(f);
            return NULL;
        }
        if (d != dim) {
            fprintf(stderr, "Dimension mismatch at vector %d: expected %d got %d\n", i, dim, d);
            free(data);
            fclose(f);
            return NULL;
        }
        if (fread(data + i*dim, sizeof(float), dim, f) != (size_t)dim) {
            fprintf(stderr, "Failed to read vector data for vector %d\n", i);
            free(data);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    *out_num_vectors = num_vectors;
    *out_dim = dim;
    return data;
}
