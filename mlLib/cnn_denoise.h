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
#define NUMBER_OF_IMAGES_IN_DATA_LOADER 1024
#define HEIGHT 600
#define WIDTH 800
#define IMAGE_SIZE (WIDTH * HEIGHT * 4) // RGB + Luminance

#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>

/* Opaque handle to CNN model */
typedef struct CNNDenoiser CNNDenoiser;

/* Optimizer type */
typedef enum {
    OPTIMIZER_SGD = 0,
    OPTIMIZER_ADAM = 1
} OptimizerType;

/* Loss type */
typedef enum {
    LOSS_MAE = 0,
    LOSS_MSE = 1,
    LOSS_LAPLACE = 2
} LossType;

/* Loss configuration with multiple losses and weights */
typedef struct {
    LossType types[3];       /* Up to 3 loss types */
    float weights[3];        /* Weight for each loss */
    int num_losses;          /* Number of active losses */
} LossConfig;

/* Configuration for network architecture */
typedef struct {
    int input_width;
    int input_height;
    int input_channels;      /* Must be multiple of 4 */
    int output_channels;     /* Must be multiple of 4 */
    float learning_rate;
    int use_profiling;       /* Enable timing profiling */
    int residual_mode;       /* 1 = predict noise (output = input - prediction), 0 = direct */
    int auto_tune_workgroup; /* Auto-tune work group sizes on first run */
    OptimizerType optimizer; /* SGD or Adam */
    LossConfig loss_config;  /* Loss function configuration */
    float adam_beta1;        /* Adam: momentum decay (default 0.9) */
    float adam_beta2;        /* Adam: RMSprop decay (default 0.999) */
    float adam_epsilon;      /* Adam: numerical stability (default 1e-8) */
} CNNConfig;

typedef struct learning_rate_decay {
    float initial_lr;
    float decay_rate;
    int decay_steps;
    int step;
} LearningRateDecay;

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

typedef struct {
    int current_index;      /* Counter for tracking iterations */
    char folder_path[512];  /* Root path containing image folders */
} DataLoader;

typedef  struct {
    float lowRes [IMAGE_SIZE];  // RGB + Luminance
    float highRes [IMAGE_SIZE];  // RGB + Luminance
} ImageSample;

void fillDataLoader(DataLoader* loader, char *folder_path);
void getNextImagePair(DataLoader* loader, ImageSample* sample);

void learning_rate_decay_init(LearningRateDecay* lr_decay, 
                              float initial_lr, float decay_rate, int decay_steps);
float learning_rate_decay_get(LearningRateDecay* lr_decay, int current_step);

/* Create/destroy network */
CNNDenoiser* cnn_create(CNNConfig config);
void cnn_destroy(CNNDenoiser* cnn);

/* Helper to create default config */
CNNConfig cnn_default_config(int width, int height, int channels);

/* Build network architecture */
int cnn_add_layer(CNNDenoiser* cnn, LayerConfig layer);
int cnn_finalize(CNNDenoiser* cnn);  /* Call after adding all layers */

/* Training */
float cnn_train_step(CNNDenoiser* cnn, 
                     float* noisy_input,   /* [batch][h][w][channels] */
                     float* clean_target,  /* [batch][h][w][channels] */
                     int batch_size);

void cnn_set_learning_rate(CNNDenoiser* cnn, float learning_rate);
float cnn_get_learning_rate(CNNDenoiser* cnn);

/* Get last forward pass output (after train step or denoise call) */
void cnn_get_output(CNNDenoiser* cnn, float* output);

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

/* Save/Load network weights */
int cnn_save_weights(CNNDenoiser* cnn, const char* filepath);
int cnn_load_weights(CNNDenoiser* cnn, const char* filepath);

/* Data helpers */
void cnn_add_gaussian_noise(float* clean, float* noisy, 
                            int size, float sigma);
float cnn_compute_psnr(float* image1, float* image2, int size);

/* ===== Easy-to-use RGB helper functions =====
 * These handle conversion between RGB (3 channels) and RGBA (RGB + Luminance, 4 channels)
 * Required because GPU uses float4 for optimal vectorization
 */

/* Convert RGB uint8 image to RGBA float (RGB + Luminance for 4-channel alignment)
 * rgb: input RGB image [height][width][3], values 0-255
 * rgba: output RGBA float [height][width][4], values 0.0-1.0, channel 3 = luminance
 */
void cnn_rgb_to_rgba_luminance(const unsigned char* rgb, float* rgba, 
                                int width, int height);

/* Load RGB + separate luminance images into RGBA float format
 * rgb: input RGB image [height][width][3], values 0-255
 * lum: input luminance image [height][width], values 0-255
 * rgba: output RGBA float [height][width][4], values 0.0-1.0, channel 3 = luminance
 */
void cnn_load_rgba_luminance(const unsigned char* rgb, const unsigned char* lum,
                              float* rgba, int width, int height);

/* Convert RGBA float back to RGB uint8 image (discards luminance channel)
 * rgba: input RGBA float [height][width][4], values 0.0-1.0
 * rgb: output RGB image [height][width][3], values 0-255
 */
void cnn_rgba_luminance_to_rgb(const float* rgba, unsigned char* rgb, 
                                int width, int height);

/* Prepare training batch: Load RGB+Luminance -> RGBA, add noise, clamp values
 * clean_rgb: input clean RGB image 800x600x3 (uint8)
 * clean_lum: input clean luminance image 800x600 (uint8)
 * noisy_rgb: output noisy RGB image 800x600x3 (uint8, optional - can be NULL)
 * noisy_lum: output noisy luminance image 800x600 (uint8, optional - can be NULL)
 * clean_rgba: output clean RGBA float 800x600x4 (for training target)
 * noisy_rgba: output noisy RGBA float 800x600x4 (for training input)
 * noise_sigma: Gaussian noise level (e.g., 0.05 for 5% noise)
 * Returns: 0 on success, -1 on error
 */
int cnn_prepare_training_batch(const unsigned char* clean_rgb,
                                const unsigned char* clean_lum,
                                unsigned char* noisy_rgb,
                                unsigned char* noisy_lum,
                                float* clean_rgba, float* noisy_rgba, 
                                int width, int height, float noise_sigma);

/* Easy inference: RGB image in, denoised RGB image out
 * input_rgb: noisy RGB image 800x600x3 (uint8)
 * output_rgb: denoised RGB image 800x600x3 (uint8)
 * Returns: 0 on success, -1 on error
 * Note: Internally converts to RGBA, runs network, converts back to RGB
 */
int cnn_inference_rgb(CNNDenoiser* cnn, const unsigned char* input_rgb, 
                      unsigned char* output_rgb, int width, int height);

/* ===== Logging Functions for wandb/logger.py ===== */

/* Convert RGB data to base64 (no allocation, buffer must be pre-allocated) */
size_t rgb_to_base64_noalloc(
    const unsigned char *data,
    size_t len,
    char *out
);

/* Convert float RGBA image to base64 (buffers must be pre-allocated) */
void imageToBase64_noalloc(
    const float *image,
    int width,
    int height,
    unsigned char *rgb_buffer,
    char *base64_buffer
);

/* Convert planar RGBA layout to interleaved RGBA layout */
void planarToInterleaved(
    const float *planar,
    float *interleaved,
    int width,
    int height,
    int channels
);

/* Convert interleaved RGBA layout to planar RGBA layout */
void interleavedToPlanar(
    const float *interleaved,
    float *planar,
    int width,
    int height,
    int channels
);

/* Send images to Python logger endpoint */
int send_images_to_python(
    const char *url,
    const char *input_img_b64,
    const char *original_img_b64,
    const char *prediction_img_b64,
    int step
);

/* Send training metadata to Python logger endpoint */
void send_metadata_to_python(
    const char *url,
    int step,
    float loss,
    float learning_rate,
    float timeTookms
);

#ifdef __cplusplus
}
#endif

#endif /* CNN_DENOISE_H */
