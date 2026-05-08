"""
PairAlign: A Framework for Sequence Tokenization via Self-Alignment with Applications to Audio Tokenization
Source Paper: http://arxiv.org/abs/2605.06582v1

Mathematical Idea:
The PairAlign framework proposes learning a tokenizer by leveraging a self-alignment mechanism.
It aims to map a continuous input sequence (e.g., audio features) into a sequence of discrete
tokens. The core idea is to train an encoder-quantizer-decoder pipeline such that the original
input sequence `X` can be effectively "reconstructed" from its discrete token representation `T`.

The learning objective is primarily an alignment loss between the original input sequence `X`
and a reconstructed sequence `X_hat` (derived from `T`). This alignment loss is typically
computed using a differentiable pairwise alignment algorithm, such as Soft-Dynamic Time Warping (Soft-DTW).
By minimizing this alignment cost, the model is encouraged to learn tokens that preserve the
temporal structure and content of the original sequence, even after compression and quantization.

The framework consists of:
1.  An `Encoder` (f_enc): Maps the input sequence `X` to a sequence of continuous embeddings `Z`.
2.  A `Quantizer` (f_quant): Maps `Z` to discrete tokens `T` and corresponding quantized embeddings `Z_q`.
    A Vector Quantizer (VQ-VAE style) is commonly used, involving a learnable codebook.
3.  A `Decoder` (f