#include "faiss_wrapper.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFSearch.h>
#include <faiss/Index.h>

using namespace faiss;

extern "C" {

FaissIndex create_flat_index(int dim) {
    return (FaissIndex)(new IndexFlatL2(dim));
}

FaissIndex create_hnsw_index(int dim, int M) {
    return (FaissIndex)(new IndexHNSWFlat(dim, M));
}

FaissIndex create_ivf_flat_index(int dim, int nlist) {
    IndexFlatL2* quantizer = new IndexFlatL2(dim);
    IndexIVFFlat* ivf = new IndexIVFFlat(quantizer, dim, nlist, METRIC_L2);
    return (FaissIndex)ivf;
}

FaissIndex create_ivf_hnsw_index(int dim, int nlist, int M) {
    IndexHNSWFlat* quantizer = new IndexHNSWFlat(dim, M);
    IndexIVFFlat* ivf = new IndexIVFFlat(quantizer, dim, nlist, METRIC_L2);
    return (FaissIndex)ivf;
}

FaissIndex create_ivf_pq_index(int dim, int nlist, int m, int nbits) {
    IndexFlatL2* quantizer = new IndexFlatL2(dim);
    IndexIVFPQ* ivfpq = new IndexIVFPQ(quantizer, dim, nlist, m, nbits);
    return (FaissIndex)ivfpq;
}

void train_ivf_index(FaissIndex index, float* data, int n) {
    IndexIVF* ivf = dynamic_cast<IndexIVF*>((Index*)index);
    if (ivf) {
        ivf->train(n, data);
    }
}

void add_vectors(FaissIndex index, float* data, int n) {
    ((Index*)index)->add(n, data);
}

void search_vectors(FaissIndex index, float* queries, int n_query, int k, int* labels, float* distances) {
    ((Index*)index)->search(n_query, queries, k, distances, labels);
}

void free_faiss_index(FaissIndex index) {
    delete (Index*)index;
}

}
