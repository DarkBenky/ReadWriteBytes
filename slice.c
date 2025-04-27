#include <stdlib.h>
#include <string.h>
#include "slice.h"

void slice_init(struct Slice *slice, int capacity) {
    slice->data = malloc(capacity);
    slice->length = 0;
    slice->capacity = capacity;
}

void clear_slice(struct Slice *slice) {
    slice->length = 0;
}

void slice_push(struct Slice *slice, void *data, int size) {
    if (slice->length + size > slice->capacity) {
        slice->capacity = (slice->length + size) * 2;
        slice->data = realloc(slice->data, slice->capacity);
    }
    memcpy((char *)slice->data + slice->length, data, size);
    slice->length += size;
}

void slice_pop(struct Slice *slice, void *data, int size) {
    if (slice->length < size) {
        return;
    }
    memcpy(data, (char *)slice->data + slice->length - size, size);
    slice->length -= size;
}

void slice_free(struct Slice *slice) {
    free(slice->data);
    slice->data = NULL;
    slice->length = 0;
    slice->capacity = 0;
}
