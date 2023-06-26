#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <stdexcept>
#include <vector>
#include "inference_context.h"

template <typename T>
void launch_apply_rotary_pos_emb(T *mixed_query,
                                 T *key_layer,
                                 unsigned head_size,
                                 unsigned seq_len,
                                 unsigned rotary_dim,
                                 unsigned offset,
                                 unsigned num_heads,
                                 unsigned batch,
                                 cudaStream_t stream,
                                 int max_out_tokens);

std::vector<at::Tensor> apply_rotary_pos_emb(at::Tensor &mixed_query,
                                             at::Tensor &key_layer,
                                             int64_t rotary_dim,
                                             int64_t offset,
                                             int64_t num_heads,
                                             bool rotate_half,
                                             int64_t max_token_length)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto query_cont = mixed_query.contiguous();
    auto key_cont = key_layer.contiguous();

    unsigned bsz = mixed_query.size(0);
    unsigned head_size = mixed_query.size(2) / num_heads;
    unsigned seq_len = mixed_query.size(1);

    if (mixed_query.scalar_type() == at::kFloat)
        launch_apply_rotary_pos_emb<float>((float *)query_cont.data_ptr(),
                                           (float *)key_cont.data_ptr(),
                                           head_size,
                                           seq_len,
                                           rotary_dim,
                                           offset,
                                           num_heads,
                                           bsz,
                                           stream,
                                           max_token_length);
    else
        launch_apply_rotary_pos_emb<__half>((__half *)query_cont.data_ptr(),
                                            (__half *)key_cont.data_ptr(),
                                            head_size,
                                            seq_len,
                                            rotary_dim,
                                            offset,
                                            num_heads,
                                            bsz,
                                            stream,
                                            max_token_length);
    return {query_cont, key_cont};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("apply_rotary_pos_emb", &apply_rotary_pos_emb, "DS rotary pos emb wrapper");
}

TORCH_LIBRARY(ds, m)
{
    m.def("apply_rotary_pos_emb", apply_rotary_pos_emb);
}