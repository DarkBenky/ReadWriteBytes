#include "../cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 16
#define HEIGHT 16
#define CHANNELS 4
#define IMAGE_SIZE (WIDTH * HEIGHT * CHANNELS)

int main() {
	printf("=== Testing Laplacian Loss with Batch Size 1 ===\n\n");

	CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, CHANNELS);
	cfg.max_batch_size = 32;
	cfg.optimizer = OPTIMIZER_ADAM;
	cfg.learning_rate = 0.001f;

	cfg.loss_config.num_losses = 1;
	cfg.loss_config.types[0] = LOSS_LAPLACE;
	cfg.loss_config.weights[0] = 1.0f;

	CNNDenoiser *cnn = cnn_create(cfg);

	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 8, 1, -1, -1, "conv1"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 8, 4, 0, -1, -1, "output"});

	cnn_finalize(cnn);

	float *input = malloc(IMAGE_SIZE * sizeof(float));
	float *target = malloc(IMAGE_SIZE * sizeof(float));

	printf("Creating test pattern with edges...\n");
	for (int c = 0; c < CHANNELS; c++) {
		for (int y = 0; y < HEIGHT; y++) {
			for (int x = 0; x < WIDTH; x++) {
				int idx = c * HEIGHT * WIDTH + y * WIDTH + x;

				if (c < 3) {
					if (x < WIDTH / 2) {
						input[idx] = 0.2f;
						target[idx] = 0.3f;
					} else {
						input[idx] = 0.8f;
						target[idx] = 0.9f;
					}
				} else {
					input[idx] = 0.5f;
					target[idx] = 0.5f;
				}
			}
		}
	}

	printf("Input pattern: left half=0.2, right half=0.8 (RGB only)\n");
	printf("Target pattern: left half=0.3, right half=0.9 (RGB only)\n\n");

	printf("--- Test 1: Training step with batch_size=1 ---\n");
	float loss = cnn_train_step(cnn, input, target, 1);

	float mae, mse, laplace, color, ssim, sobel;
	cnn_get_individual_losses(cnn, &mae, &mse, &laplace, &color, &ssim, &sobel);

	printf("Returned total loss: %.10f\n", loss);
	printf("Laplacian loss: %.10f\n", laplace);
	printf("MAE: %.10f\n", mae);
	printf("MSE: %.10f\n", mse);

	if (fabs(laplace) < 1e-9f) {
		printf("\n*** FAIL: Laplacian loss is ZERO! ***\n");
		printf("This indicates the Laplacian kernel is not computing properly.\n\n");

		printf("Checking output values...\n");
		float *output = malloc(IMAGE_SIZE * sizeof(float));
		cnn_get_output(cnn, output);

		printf("First 20 output values (RGB channels):\n");
		for (int i = 0; i < 20; i++) {
			printf("%.4f ", output[i]);
			if ((i + 1) % 10 == 0) printf("\n");
		}

		printf("\nComputing Laplacian manually for verification...\n");
		float manual_laplace = 0.0f;
		int pixel_count = 0;

		for (int c = 0; c < 3; c++) {
			for (int y = 1; y < HEIGHT - 1; y++) {
				for (int x = 1; x < WIDTH - 1; x++) {
					int idx = c * HEIGHT * WIDTH + y * WIDTH + x;

					float lap_out = -4.0f * output[idx] +
									output[idx - 1] + output[idx + 1] +
									output[idx - WIDTH] + output[idx + WIDTH];

					float lap_tgt = -4.0f * target[idx] +
									target[idx - 1] + target[idx + 1] +
									target[idx - WIDTH] + target[idx + WIDTH];

					float diff = fabsf(lap_out - lap_tgt);
					manual_laplace += diff;
					pixel_count++;
				}
			}
		}

		int rgb_pixels = WIDTH * HEIGHT * 3;
		manual_laplace /= rgb_pixels;

		printf("Manual Laplacian computation:\n");
		printf("  Interior pixels checked: %d\n", pixel_count);
		printf("  Total RGB pixels: %d\n", rgb_pixels);
		printf("  Normalized manual Laplacian: %.10f\n", manual_laplace);

		free(output);
	} else {
		printf("\n*** SUCCESS: Laplacian loss is non-zero! ***\n");
		printf("Laplacian loss is working correctly.\n");
	}

	printf("\n--- Test 2: Multiple training steps ---\n");
	for (int i = 0; i < 3; i++) {
		loss = cnn_train_step(cnn, input, target, 1);
		cnn_get_individual_losses(cnn, &mae, &mse, &laplace, &color, &ssim, &sobel);
		printf("Step %d - Loss: %.6f, Laplace: %.6f\n", i + 1, loss, laplace);
	}

	free(input);
	free(target);
	cnn_destroy(cnn);

	return 0;
}
