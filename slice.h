#ifndef SLICE_H
#define SLICE_H

struct Slice {
    void *data;
    int length;
    int capacity;
};

void slice_init(struct Slice *slice, int capacity);
void clear_slice(struct Slice *slice);
void slice_push(struct Slice *slice, void *data, int size);
void slice_pop(struct Slice *slice, void *data, int size);
void slice_free(struct Slice *slice);

#endif
