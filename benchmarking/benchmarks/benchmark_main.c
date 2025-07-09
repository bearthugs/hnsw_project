#include <stdio.h>
void run_faiss_flat_benchmark(const char*, const char*, const char*, int);

int main() {
    const char* base_path = "datasets/sift1m_base.fvecs";
    const char* query_path = "datasets/sift1m_query.fvecs";
    const char* gt_path = "datasets/sift1m_groundtruth.ivecs";
    int k = 10;

    run_faiss_flat_benchmark(base_path, query_path, gt_path, k);
    return 0;
}
