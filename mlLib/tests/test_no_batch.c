#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
	printf("=== Quick Test: No Batch Mode with Residual Layers ===\n\n");

	CNNConfig cfg = cnn_default_config(WIDTH, HEIGHT, 4);
	cfg.max_batch_size = 1; /* No batch mode */
	cfg.learning_rate = 0.001f;
	cfg.optimizer = OPTIMIZER_ADAM;
	cfg.residual_mode = 0;

	cfg.loss_config.num_losses = 1;
	cfg.loss_config.types[0] = LOSS_MAE;
	cfg.loss_config.weights[0] = 1.0f;

	CNNDenoiser *cnn = cnn_create(cfg);

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

	int size = WIDTH * HEIGHT * 4;
	float *input = malloc(size * sizeof(float));
	float *target = malloc(size * sizeof(float));
	float *output = malloc(size * sizeof(float));

	/* Realistic test: varied values */
	for (int i = 0; i < size; i++) {
		float v = (i % 256) / 255.0f;
		input[i] = v * 0.5f + 0.3f;	 /* 0.3 to 0.8 */
		target[i] = v * 0.4f + 0.2f; /* 0.2 to 0.6 */
	}

	printf("\nTraining 20 steps (batch_size=1)...\n");
	for (int step = 0; step < 20; step++) {
		float loss = cnn_train_step(cnn, input, target, 1);
		if (step % 5 == 0) {
			cnn_get_output(cnn, output);

			/* Check if output is all zeros */
			int zeros = 0;
			float sum = 0;
			for (int i = 0; i < size; i++) {
				if (output[i] == 0.0f) zeros++;
				sum += output[i];
			}

			printf("Step %2d: Loss=%.6f, Avg Output=%.6f, Zeros=%d/%d (%.1f%%)\n",
				   step, loss, sum / size, zeros, size, 100.0f * zeros / size);
		}
	}

	cnn_get_output(cnn, output);

	/* Final check */
	int zero_count = 0;
	float min = 1e9, max = -1e9, sum = 0;
	for (int i = 0; i < size; i++) {
		if (output[i] == 0.0f) zero_count++;
		if (output[i] < min) min = output[i];
		if (output[i] > max) max = output[i];
		sum += output[i];
	}

	printf("\nFinal output stats:\n");
	printf("  min=%.6f, max=%.6f, avg=%.6f\n", min, max, sum / size);
	printf("  zeros=%d/%d (%.1f%%)\n", zero_count, size, 100.0f * zero_count / size);
	printf("  First 5 pixels: %.6f %.6f %.6f %.6f %.6f\n",
		   output[0], output[1], output[2], output[3], output[4]);

	if (zero_count == size) {
		printf("\n❌ FAILED: All outputs are zero!\n");
	} else if (zero_count > size / 2) {
		printf("\n⚠️  WARNING: More than 50%% zeros\n");
	} else {
		printf("\n✅ SUCCESS: Network is producing varied output!\n");
	}

	free(input);
	free(target);
	free(output);
	cnn_destroy(cnn);

	return 0;
}
