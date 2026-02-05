#include "cnn_denoise.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
	CNNConfig cfg = cnn_default_config(64, 64, 4);
	cfg.use_profiling = 1;
	cfg.max_batch_size = 16;

	CNNDenoiser *cnn = cnn_create(cfg);
	if (!cnn) {
		fprintf(stderr, "Failed to create CNN\n");
		return 1;
	}

	LayerConfig layer = {0};
	layer.type = LAYER_RESIDUAL_INPUT;
	layer.cin = 4;
	layer.cout = 4;
	layer.use_relu = 0;
	layer.skip_from = -1;
	layer.residual_from = -1;
	strcpy(layer.name, "save_input");
	cnn_add_layer(cnn, layer);

	layer.type = LAYER_CONV;
	layer.cin = 4;
	layer.cout = 8;
	layer.use_relu = 1;
	strcpy(layer.name, "conv1");
	cnn_add_layer(cnn, layer);

	layer.cin = 8;
	layer.cout = 4;
	layer.use_relu = 0;
	strcpy(layer.name, "conv2");
	cnn_add_layer(cnn, layer);

	layer.type = LAYER_RESIDUAL_SUBTRACT;
	layer.cin = 4;
	layer.cout = 4;
	strcpy(layer.name, "denoise");
	cnn_add_layer(cnn, layer);

	if (cnn_finalize(cnn) != 0) {
		fprintf(stderr, "Failed to finalize CNN\n");
		return 1;
	}

	printf("Network created: 64x64x4, 4 layers\n");

	int batch_size = 8;
	int img_size = 64 * 64 * 4;
	float *noisy = malloc(batch_size * img_size * sizeof(float));
	float *clean = malloc(batch_size * img_size * sizeof(float));

	for (int i = 0; i < batch_size * img_size; i++) {
		noisy[i] = (float)rand() / RAND_MAX;
		clean[i] = (float)rand() / RAND_MAX;
	}

	printf("\nRunning 5 training steps with batch_size=%d...\n", batch_size);
	for (int step = 0; step < 5; step++) {
		float loss = cnn_train_step(cnn, noisy, clean, batch_size);
		TimingStats stats;
		cnn_get_timing_stats(cnn, &stats);

		printf("Step %d: Forward %.2f ms, Backward %.2f ms, Loss %.2f ms, Update %.2f ms, Total %.2f ms (Loss: %.6f)\n",
			   step, stats.forward_time_ms, stats.backward_time_ms,
			   stats.loss_time_ms, stats.update_time_ms, stats.total_time_ms, loss);
	}

	free(noisy);
	free(clean);
	cnn_destroy(cnn);

	printf("\n=== Test complete ===\n");
	return 0;
}
