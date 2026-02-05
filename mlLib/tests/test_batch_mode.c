#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
	printf("=== Testing Batch Training (reproducing black output) ===\n\n");

	/* Use same settings as train.c */
	CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, 4);
	cfg.max_batch_size = 32;
	cfg.learning_rate = 0.00035f;
	cfg.optimizer = OPTIMIZER_ADAM;
	cfg.adam_beta1 = 0.95f;
	cfg.adam_beta2 = 0.999f;
	cfg.residual_mode = 0;

	cfg.loss_config.num_losses = 3;
	cfg.loss_config.types[0] = LOSS_MAE;
	cfg.loss_config.weights[0] = 1.5f;
	cfg.loss_config.types[1] = LOSS_SSIM;
	cfg.loss_config.weights[1] = 1.5f;
	cfg.loss_config.types[2] = LOSS_SOBEL;
	cfg.loss_config.weights[2] = 0.5f;

	CNNDenoiser *cnn = cnn_create(cfg);

	/* Same architecture as train.c */
	cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_INPUT, 4, 4, 0, -1, -1, "save_input"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 16, 1, -1, -1, "noise_1"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 20, 1, -1, -1, "noise_2"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 20, 24, 1, -1, -1, "noise_3"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 24, 20, 1, -1, -1, "noise_4"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 20, 16, 1, -1, -1, "noise_5"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 4, 0, -1, -1, "noise_out"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_SUBTRACT, 4, 4, 0, -1, 0, "denoise"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 12, 1, -1, -1, "refine_1"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 12, 8, 1, -1, -1, "refine_2"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 8, 4, 0, -1, -1, "output"});

	cnn_finalize(cnn);
	cnn_print_architecture(cnn);

	int img_size = WIDTH * HEIGHT * 4;
	const int BATCH_SIZE = 16;

	float *batch_input = malloc(BATCH_SIZE * img_size * sizeof(float));
	float *batch_target = malloc(BATCH_SIZE * img_size * sizeof(float));
	float *output = malloc(img_size * sizeof(float));

	printf("\n=== Test 1: Constant batch ===\n");
	/* Fill batch with constant values */
	for (int b = 0; b < BATCH_SIZE; b++) {
		for (int i = 0; i < img_size; i++) {
			batch_input[b * img_size + i] = 0.8f;
			batch_target[b * img_size + i] = 0.5f;
		}
	}

	printf("Training 1 step with batch_size=%d...\n", BATCH_SIZE);
	float loss = cnn_train_step(cnn, batch_input, batch_target, BATCH_SIZE);
	printf("Loss: %.6f (expected ~0.3)\n", loss);

	cnn_get_batch_output(cnn, output, 0);
	printf("First image output[0]: %.6f (target: 0.5)\n", output[0]);
	printf("First image output[1000]: %.6f\n", output[1000]);

	int zero_count = 0;
	float min = 1e9, max = -1e9, sum = 0;
	for (int i = 0; i < img_size; i++) {
		if (output[i] == 0.0f) zero_count++;
		if (output[i] < min) min = output[i];
		if (output[i] > max) max = output[i];
		sum += output[i];
	}
	printf("Output stats: min=%.6f, max=%.6f, avg=%.6f, zeros=%d/%d (%.1f%%)\n",
		   min, max, sum / img_size, zero_count, img_size, 100.0f * zero_count / img_size);

	/* Train more */
	printf("\nTraining 10 more steps...\n");
	for (int step = 0; step < 10; step++) {
		loss = cnn_train_step(cnn, batch_input, batch_target, BATCH_SIZE);
		if (step % 2 == 0) {
			cnn_get_batch_output(cnn, output, 0);
			printf("Step %2d: Loss=%.6f, Output[0]=%.6f\n", step, loss, output[0]);
		}
	}

	cnn_get_batch_output(cnn, output, 0);
	printf("\nFinal output[0]: %.6f (should be closer to 0.5)\n", output[0]);

	/* Test with single image for comparison */
	printf("\n=== Test 2: Single image (batch_size=1) ===\n");
	float *single_input = malloc(img_size * sizeof(float));
	float *single_target = malloc(img_size * sizeof(float));

	for (int i = 0; i < img_size; i++) {
		single_input[i] = 0.8f;
		single_target[i] = 0.5f;
	}

	/* Create new network */
	cnn_destroy(cnn);
	cfg.max_batch_size = 1; /* Disable batch mode */
	cnn = cnn_create(cfg);
	cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_INPUT, 4, 4, 0, -1, -1, "save_input"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 16, 1, -1, -1, "noise_1"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 20, 1, -1, -1, "noise_2"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 20, 24, 1, -1, -1, "noise_3"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 24, 20, 1, -1, -1, "noise_4"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 20, 16, 1, -1, -1, "noise_5"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 16, 4, 0, -1, -1, "noise_out"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_RESIDUAL_SUBTRACT, 4, 4, 0, -1, 0, "denoise"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 4, 12, 1, -1, -1, "refine_1"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 12, 8, 1, -1, -1, "refine_2"});
	cnn_add_layer(cnn, (LayerConfig){LAYER_CONV, 8, 4, 0, -1, -1, "output"});
	cnn_finalize(cnn);

	printf("Training 10 steps with batch_size=1...\n");
	for (int step = 0; step < 10; step++) {
		loss = cnn_train_step(cnn, single_input, single_target, 1);
		if (step % 2 == 0) {
			cnn_get_output(cnn, output);
			printf("Step %2d: Loss=%.6f, Output[0]=%.6f\n", step, loss, output[0]);
		}
	}

	cnn_get_output(cnn, output);
	printf("\nFinal output[0]: %.6f (single image mode)\n", output[0]);

	printf("\n=== Comparison ===\n");
	printf("If batch mode gives all zeros or huge losses: bug in batch kernels\n");
	printf("If both modes work: issue with real image data loading\n");

	free(batch_input);
	free(batch_target);
	free(single_input);
	free(single_target);
	free(output);
	cnn_destroy(cnn);

	return 0;
}
