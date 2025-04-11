#ifndef EVOMORPH_H
#define EVOMORPH_H
#include "kickstart.h"

typedef void (*activation)(double* const);
typedef void (*loss)(double* const, const double* const);
typedef void (*bias_init)(double* const);
typedef void (*weight_init)(double* const);

typedef struct layer layer;
typedef struct layer {
	layer** prev;
	uint64_t prev_count;
	uint64_t prev_capacity;
	layer** next;
	uint64_t next_count;
	uint64_t next_capacity;
	union {
		struct {
			double* output;
			double** weights;
			double* bias;
			activation_function activation;
			uint64_t width;
		} layer;
		struct {
			double* output;
			uint64_t width;
		} input;
	} data;
	enum {
		LAYER_NODE,
		INPUT_NODE,
	} tag;
} layer;

typedef struct network {
	pool* mem;
	layer* input;
	layer* output;
	loss_function loss;
	bias_init bias;
	weight_init weight;
} network;

network network_init(pool* const mem);
layer* input_init(pool* const mem, uint64_t width);
layer* layer_init(pool* const mem, uint64_t width);
void layer_link(pool* const mem, layer* const a, layer* const b);
void layer_unlink(layer* const a, layer* const b);
void layer_insert(pool* const mem, layer* const a, layer* const b, layer* const c);

void forward(layer* const node);
void backward(layer* const node);

#endif
