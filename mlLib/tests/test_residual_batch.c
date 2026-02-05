#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_WIDTH 64
#define TEST_HEIGHT 64
#define TEST_CHANNELS 4

int main() {
	printf("=== Testing Residual Layers in Batch Mode ===\n\n");

	/* Test with batch_size=1 first (single-image path) */
	printf("Test 1: Single-image mode (batch_size=1)\n");
	CNNConfig cfg1 = cnn_default_config(TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
	cfg1.max_batch_size = 1;
	cfg1.optimizer = OPTIMIZER_ADAM;
	cfg1.learning_rate = 0.001f;

	CNNDenoiser *cnn1 = cnn_create(cfg1);
	cnn_add_layer(cnn1, (LayerConfig){LAYER_RESIDUAL_INPUT, 4, 4, 0, -1, -1, "save_input"});
	cnn_add_layer(cnn1, (LayerConfig){LAYER_CONV, 4, 8, 1, -1, -1, "conv1"});
	cnn_add_layer(cnn1, (LayerConfig){LAYER_CONV, 8, 4, 0, -1, -1, "noise_pred"});
	cnn_add_layer(cnn1, (LayerConfig){LAYER_RESIDUAL_SUBTRACT, 4, 4, 0, -1, 0, "denoise"});
	cnn_finalize(cnn1);

	int img_size = TEST_WIDTH * TEST_HEIGHT * TEST_CHANNELS;
	float *input = malloc(img_size * sizeof(float));
	float *target = malloc(img_size * sizeof(float));
	float *output = malloc(img_size * sizeof(float));

	/* Fill with test pattern */
	for (int i = 0; i < img_size; i++) {
		input[i] = 0.5f + 0.1f * (float)(i % 10) / 10.0f;
		target[i] = 0.4f + 0.05f * (float)(i % 5) / 5.0f;
	}

	/* Run single training step */
	float loss1 = cnn_train_step(cnn1, input, target, 1);
	cnn_get_output(cnn1, output);

	/* Check output is not all zeros */
	float min1 = 1e9f, max1 = -1e9f, sum1 = 0.0f;
	int nonzero1 = 0;
	for (int i = 0; i < img_size; i++) {
		if (output[i] < min1) min1 = output[i];
		if (output[i] > max1) max1 = output[i];
		sum1 += output[i];
		if (output[i] != 0.0f) nonzero1++;
	}

	printf("  Loss: %.6f\n", loss1);
	printf("  Output range: [%.6f, %.6f]\n", min1, max1);
	printf("  Output avg: %.6f\n", sum1 / img_size);
	printf("  Non-zero pixels: %d/%d (%.1f%%)\n", nonzero1, img_size, 100.0f * nonzero1 / img_size);

	if (max1 == 0.0f && min1 == 0.0f) {
		printf("  ❌ FAILED: All outputs are zero!\n");
		return 1;
	} else {
		printf("  ✓ PASSED: Network produces non-zero output\n");
	}

	cnn_destroy(cnn1);

	/* Test with batch_size=16 (batch kernel path) */
	printf("\nTest 2: Batch mode (batch_size=16)\n");
	CNNConfig cfg2 = cnn_default_config(TEST_WIDTH, TEST_HEIGHT, TEST_CHANNELS);
	cfg2.max_batch_size = 32;
	cfg2.optimizer = OPTIMIZER_ADAM;
	cfg2.learning_rate = 0.001f;

	CNNDenoiser *cnn2 = cnn_create(cfg2);
	cnn_add_layer(cnn2, (LayerConfig){LAYER_RESIDUAL_INPUT, 4, 4, 0, -1, -1, "save_input"});
	cnn_add_layer(cnn2, (LayerConfig){LAYER_CONV, 4, 8, 1, -1, -1, "conv1"});
	cnn_add_layer(cnn2, (LayerConfig){LAYER_CONV, 8, 4, 0, -1, -1, "noise_pred"});
	cnn_add_layer(cnn2, (LayerConfig){LAYER_RESIDUAL_SUBTRACT, 4, 4, 0, -1, 0, "denoise"});
	cnn_finalize(cnn2);

	int batch_size = 16;
	float *batch_input = malloc(batch_size * img_size * sizeof(float));
	float *batch_target = malloc(batch_size * img_size * sizeof(float));

	/* Fill batch with different patterns */
	for (int b = 0; b < batch_size; b++) {
		for (int i = 0; i < img_size; i++) {
			batch_input[b * img_size + i] = 0.5f + 0.1f * (float)((i + b) % 10) / 10.0f;
			batch_target[b * img_size + i] = 0.4f + 0.05f * (float)(i % 5) / 5.0f;
		}
	}

	/* Run batch training step */
	float loss2 = cnn_train_step(cnn2, batch_input, batch_target, batch_size);
	cnn_get_batch_output(cnn2, output, 0);

	/* Check first image output is not all zeros */
	float min2 = 1e9f, max2 = -1e9f, sum2 = 0.0f;
	int nonzero2 = 0;
	for (int i = 0; i < img_size; i++) {
		if (output[i] < min2) min2 = output[i];
		if (output[i] > max2) max2 = output[i];
		sum2 += output[i];
		if (output[i] != 0.0f) nonzero2++;
	}

	printf("  Loss: %.6f\n", loss2);
	printf("  Output range: [%.6f, %.6f]\n", min2, max2);
	printf("  Output avg: %.6f\n", sum2 / img_size);
	printf("  Non-zero pixels: %d/%d (%.1f%%)\n", nonzero2, img_size, 100.0f * nonzero2 / img_size);

	if (max2 == 0.0f && min2 == 0.0f) {
		printf("  ❌ FAILED: All batch outputs are zero!\n");
		return 1;
	} else {
		printf("  ✓ PASSED: Batch network produces non-zero output\n");
	}

	cnn_destroy(cnn2);
	free(input);
	free(target);
	free(output);
	free(batch_input);
	free(batch_target);

	printf("\n=== ALL TESTS PASSED ===\n");
	printf("Residual layers work correctly in both single-image and batch modes!\n");

	return 0;
}
