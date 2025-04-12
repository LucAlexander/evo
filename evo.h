#ifndef EVOMORPH_H
#define EVOMORPH_H
#include "kickstart.h"

#define TEMP_POOL_SIZE 0x1000000

typedef void (*activation_function)(double* const, const double* const, uint64_t);
typedef void (*loss_function)(double* const, const double* const, uint64_t);
typedef void (*loss_derivative)(double* const, const double* const, const double* const, uint64_t);
typedef void (*bias_init)(double* const, uint64_t);
typedef void (*weight_init)(double* const, uint64_t);

typedef struct layer layer;
typedef struct layer {
	layer** prev;
	uint64_t prev_count;
	uint64_t prev_capacity;
	layer** next;
	uint64_t next_count;
	uint64_t next_capacity;
	uint64_t pass_index;
	union {
		struct {
			uint64_t width;
			double* output;
			double* activated;
			// width x number of input weights, in same order as layer** prev
			double** weights;
			double* bias;
			activation_function activation;
			activation_function derivative;
			double** weight_gradients;
			double* bias_gradients;
			double* activation_gradients;
		} layer;
		struct {
			uint64_t width;
			double* output;
		} input;
	} data;
	enum {
		LAYER_NODE,
		INPUT_NODE,
	} tag;
	uint8_t simulated;
} layer;

typedef struct network {
	pool* mem;
	pool temp;
	layer* input;
	layer* output;
	loss_function loss;
	loss_derivative derivative;
	double* loss_output;
	bias_init bias;
	weight_init weight;
	uint64_t batch_size;
	double learning_rate;
} network;

network network_init(pool* const mem, uint64_t input, uint64_t output, weight_init w, bias_init b, uint64_t batch_size, double learning_rate, loss_function l, loss_derivative ld);
layer* input_init(pool* const mem, uint64_t width);
layer* layer_init(pool* const mem, uint64_t width);
void layer_link(pool* const mem, layer* const a, layer* const b);
void layer_unlink(layer* const a, layer* const b);
void layer_insert(pool* const mem, layer* const a, layer* const b, layer* const c);
void reset_simulation_flags(layer* const node);
void allocate_weights(pool* const mem, layer* const node, uint64_t pass_index);
void forward(layer* const node, uint64_t pass_index);
void backward(network* const net, layer* const node, uint64_t pass_index);
void apply_gradients(network* const net, layer* const node, uint64_t pass_index);
void zero_gradients(layer* const node, uint64_t pass_index);
void network_train(network* const net, double** data, uint64_t data_size, double** expected);

#endif
