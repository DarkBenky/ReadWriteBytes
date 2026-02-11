#include "app.h"

int main(void) {
	struct AppState state = {0};

	if (!initializeApp(&state)) {
		return 1;
	}

	while (!appShouldExit(&state)) {
		runAppFrame(&state);
	}

	cleanupApp(&state);
	return 0;
}
