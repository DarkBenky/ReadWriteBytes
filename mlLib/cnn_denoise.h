/* cnn_denoise.h - Fast OpenCL CNN Denoising Library
 * 
 * Features:
 * - Vectorized float4 operations for 4-channel processing
 * - Encoder-decoder architecture with bottleneck
 * - GPU-accelerated training and inference
 * - Non-square image support (e.g., 800x600)
 */

#ifndef CNN_DENOISE_H
#define CNN_DENOISE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>

/* Opaque handle to CNN model */
typedef struct CNNDenoiser CNNDenoiser;

/* Configuration for network architecture */
typedef struct {
    int input_width;
    int input_height;
    int input_channels;      /* Must be multiple of 4 */
    int output_channels;     /* Must be multiple of 4 */
    float learning_rate;
    int use_profiling;       /* Enable timing profiling */
} CNNConfig;

/* Layer configuration */
typedef struct {
    int cin;                 /* Input channels */
    int cout;                /* Output channels */
    int use_relu;            /* 1 = ReLU activation, 0 = linear */
    char name[64];           /* Layer name for debugging */
} LayerConfig;

/* Timing statistics */
typedef struct {
    double forward_time_ms;
    double backward_time_ms;
    double loss_time_ms;
    double update_time_ms;
    double total_time_ms;
} TimingStats;

/* Create/destroy network */
CNNDenoiser* cnn_create(CNNConfig config);
void cnn_destroy(CNNDenoiser* cnn);

/* Build network architecture */
int cnn_add_layer(CNNDenoiser* cnn, LayerConfig layer);
int cnn_finalize(CNNDenoiser* cnn);  /* Call after adding all layers */

/* Training */
float cnn_train_step(CNNDenoiser* cnn, 
                     float* noisy_input,   /* [batch][h][w][channels] */
                     float* clean_target,  /* [batch][h][w][channels] */
                     int batch_size);

/* Inference */
int cnn_denoise(CNNDenoiser* cnn,
                float* noisy_input,
                float* denoised_output,
                int batch_size);

/* Utilities */
int cnn_get_num_parameters(CNNDenoiser* cnn);
void cnn_get_timing_stats(CNNDenoiser* cnn, TimingStats* stats);
void cnn_reset_timing_stats(CNNDenoiser* cnn);
void cnn_print_architecture(CNNDenoiser* cnn);

/* Data helpers */
void cnn_add_gaussian_noise(float* clean, float* noisy, 
                            int size, float sigma);
float cnn_compute_psnr(float* image1, float* image2, int size);

#ifdef __cplusplus
}
#endif

#endif /* CNN_DENOISE_H */
