#ifndef FAISS_WRAPPER_H
#define FAISS_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* FaissIndex;

// Create different index types
FaissIndex create_flat_index(int dim);
FaissIndex create_hnsw_index(int dim, int M);
FaissIndex create_ivf_flat_index(int dim, int nlist);
FaissIndex create_ivf_hnsw_index(int dim, int nlist, int M);
FaissIndex create_ivf_pq_index(int dim, int nlist, int m, int nbits);

// Train IVF / IVF+PQ indexes before adding vectors
void train_ivf_index(FaissIndex index, float* data, int n);

// Add vectors and search
void add_vectors(FaissIndex index, float* data, int n);
void search_vectors(FaissIndex index, float* queries, int n_query, int k, int* labels, float* distances);

// Free index memory
void free_faiss_index(FaissIndex index);

#ifdef __cplusplus
}
#endif

#endif // FAISS_WRAPPER_H