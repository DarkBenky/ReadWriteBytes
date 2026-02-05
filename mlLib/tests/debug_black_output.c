#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print_buffer(const char *name, float *data, int n) {
	printf("%s: ", name);
	for (int i = 0; i < n && i < 10; i++) {
		printf("%.6f ", data[i]);
	}
	printf("\n");
}

void print_stats(const char *name, float *data, int size) {
	float min = 1e9, max = -1e9, sum = 0;
	int zero_count = 0;
	for (int i = 0; i < size; i++) {
		if (data[i] < min) min = data[i];
		if (data[i] > max) max = data[i];
		sum += data[i];
		if (data[i] == 0.0f) zero_count++;
	}
	printf("%s: min=%.6f, max=%.6f, avg=%.6f, zeros=%d/%d (%.1f%%)\n",
		   name, min, max, sum / size, zero_count, size, 100.0f * zero_count / size);
}

int main(void) {
	printf("=== Debugging Black Output Issue ===\n\n");

	/* Create small test case */
	const int W = 64, H = 64, C = 4;
	CNNConfig cfg = cnn_default_config(W, H, C);
	cfg.learning_rate = 0.001f;
	cfg.optimizer = OPTIMIZER_ADAM;
	cfg.residual_mode = 0;
	cfg.loss_config.num_losses = 1;
	cfg.loss_config.types[0] = LOSS_MAE;
	cfg.loss_config.weights[0] = 1.0f;

	CNNDenoiser *cnn = cnn_create(cfg);

	/* Same architecture as train.c */
	printf("Building architecture...\n");
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

	int size = W * H * C;
	float *noisy = malloc(size * sizeof(float));
	float *clean = malloc(size * sizeof(float));
	float *output = malloc(size * sizeof(float));

	/* Create realistic test data */
	printf("\n=== Test 1: Constant values (simple case) ===\n");
	for (int i = 0; i < size; i++) {
		noisy[i] = 0.8f;
		clean[i] = 0.5f;
	}

	print_stats("Input (noisy)", noisy, size);
	print_stats("Target (clean)", clean, size);

	/* Initial forward pass - use train_step with learning rate 0 to just do forward */
	printf("\nInitial output (random weights, no training):\n");
	cnn_train_step(cnn, noisy, clean, 1); /* Do one step to initialize */
	cnn_get_output(cnn, output);
	print_stats("Output", output, size);
	print_buffer("Output first 10", output, 10);

	/* Train one step */
	printf("\nTraining 1 step...\n");
	float loss = cnn_train_step(cnn, noisy, clean, 1);
	printf("Loss: %.6f\n", loss);

	cnn_get_output(cnn, output);
	print_stats("Output after 1 step", output, size);
	print_buffer("Output first 10", output, 10);

	/* Train more */
	printf("\nTraining 50 more steps...\n");
	for (int i = 0; i < 50; i++) {
		loss = cnn_train_step(cnn, noisy, clean, 1);
		if (i % 10 == 0) {
			cnn_get_output(cnn, output);
			printf("Step %2d: Loss=%.6f, Output[0]=%.6f\n", i, loss, output[0]);
		}
	}

	cnn_get_output(cnn, output);
	print_stats("Output after 50 steps", output, size);
	print_buffer("Output first 10", output, 10);

	/* Test 2: Varied input like real images */
	printf("\n\n=== Test 2: Varied input (like real images) ===\n");
	for (int i = 0; i < size; i++) {
		float base = ((i % 256) / 255.0f);
		noisy[i] = base * 0.8f + 0.1f; /* 0.1 to 0.9 range */
		clean[i] = base * 0.6f + 0.2f; /* 0.2 to 0.8 range */
	}

	print_stats("Input (noisy)", noisy, size);
	print_stats("Target (clean)", clean, size);

	printf("\nInitial output with varied input:\n");
	cnn_train_step(cnn, noisy, clean, 1);
	cnn_get_output(cnn, output);
	print_stats("Output", output, size);
	print_buffer("Output first 10", output, 10);

	/* Reset and retrain from scratch */
	cnn_destroy(cnn);
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

	printf("\nTraining 100 steps with varied input...\n");
	for (int i = 0; i < 100; i++) {
		loss = cnn_train_step(cnn, noisy, clean, 1);
		if (i % 20 == 0) {
			cnn_get_output(cnn, output);
			printf("Step %3d: Loss=%.6f, Output[0]=%.6f, Output[100]=%.6f\n",
				   i, loss, output[0], output[100]);
		}
	}

	cnn_get_output(cnn, output);
	print_stats("Final output", output, size);
	print_buffer("Final output first 10", output, 10);

	printf("\n=== Analysis ===\n");
	printf("Expected behavior: Output should gradually approach target values\n");
	printf("If output is all zeros: Problem with network architecture or forward pass\n");
	printf("If output doesn't change: Problem with gradient computation or updates\n");

	free(noisy);
	free(clean);
	free(output);
	cnn_destroy(cnn);

	return 0;
}
