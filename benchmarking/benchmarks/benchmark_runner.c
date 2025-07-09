#include <stdio.h>
#include <stdlib.h>
#include "faiss_wrapper.h"
#include "benchmark_utils.h"

void run_faiss_flat_benchmark(
    const char* base_path,
    const char* query_path,
    const char* gt_path,
    int k
) {
    int nb, dim, nq, k_gt;
    float* base = load_fvecs(base_path, &nb, &dim);
    float* queries = load_fvecs(query_path, &nq, &dim);
    int* ground_truth = load_ivecs(gt_path, &nq, &k_gt);

    FaissIndex index = create_flat_index(dim);
    add_vectors(index, base, nb);

    int* labels = malloc(sizeof(int) * nq * k);
    float* distances = malloc(sizeof(float) * nq * k);

    double start = get_time_ms();
    search_vectors(index, queries, nq, k, labels, distances);
    double end = get_time_ms();

    float recall = compute_recall(labels, ground_truth, nq, k);
    printf("[FAISS Flat] Recall@%d: %.4f | Time: %.2f ms\n", k, recall, end - start);

    free_faiss_index(index);
    free(base); free(queries); free(ground_truth);
    free(labels); free(distances);
}
