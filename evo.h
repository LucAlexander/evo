#ifndef EVOMORPH_H
#define EVOMORPH_H

#include "kickstart.h"
#include <time.h>

#define TEMP_POOL_SIZE 0x1000000
#define GELU_C 0.044715
#define SELU_LAMBDA 1.0507009873554804934193349852946
#define SELU_ALPHA 1.6732632423543772848170429916717

typedef void (*activation_function)(double* const, const double* const, uint64_t, double);
typedef double (*loss_function)(double* const, const double* const, const double* const, uint64_t, double);
typedef void (*loss_derivative)(double* const, const double* const, const double* const, uint64_t, double);
typedef void (*bias_init)(double* const, uint64_t, double, double);
typedef void (*weight_init)(double** const, uint64_t, uint64_t, double, double);

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
			double parameter_a;
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
	double loss_parameter_a;
	double weight_parameter_a;
	double weight_parameter_b;
	double bias_parameter_a;
	double bias_parameter_b;
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

void set_seed(time_t seed);
double uniform_distribution(double min, double max);
double normal_distribution(double mean, double std);

double loss_mse(double* const, const double* const, const double* const, uint64_t, double);
double loss_mae(double* const, const double* const, const double* const, uint64_t, double);
double loss_mape(double* const, const double* const, const double* const, uint64_t, double);
double loss_huber(double* const, const double* const, const double* const, uint64_t, double);
double loss_huber_modified(double* const, const double* const, const double* const, uint64_t, double);
double loss_cross_entropy(double* const, const double* const, const double* const, uint64_t, double);
double loss_hinge(double* const, const double* const, const double* const, uint64_t, double);

void activation_sigmoid(double* const, const double* const, uint64_t, double);
void activation_relu(double* const, const double* const, uint64_t, double);
void activation_tanh(double* const, const double* const, uint64_t, double);
void activation_binary_step(double* const, const double* const, uint64_t, double);
void activation_linear(double* const, const double* const, uint64_t, double);
void activation_relu_leaky(double* const, const double* const, uint64_t, double);
void activation_relu_parametric(double* const, const double* const, uint64_t, double);
void activation_elu(double* const, const double* const, uint64_t, double);
void activation_softmax(double* const, const double* const, uint64_t, double);
void activation_swish(double* const, const double* const, uint64_t, double);
void activation_gelu(double* const, const double* const, uint64_t, double);
void activation_selu(double* const, const double* const, uint64_t, double);

void bias_initialization_zero(double* const, uint64_t, double, double);
void bias_initialization_const_flat(double* const, uint64_t, double, double);
void bias_initialization_const_uneven(double* const, uint64_t, double, double);

void weight_initialization_xavier(double** const, uint64_t, uint64_t, double, double);
void weight_initialization_he(double** const, uint64_t, uint64_t, double, double);
void weight_initialization_lecun(double** const, uint64_t, uint64_t, double, double);
void weight_initialization_uniform(double** const, uint64_t, uint64_t, double, double);
void weight_initialization_normal(double** const, uint64_t, uint64_t, double, double);

void loss_mse_partial(double* const, const double* const, const double* const, uint64_t, double);
void loss_mae_partial(double* const, const double* const, const double* const, uint64_t, double);
void loss_mape_partial(double* const, const double* const, const double* const, uint64_t, double);
void loss_huber_partial(double* const, const double* const, const double* const, uint64_t, double);
void loss_huber_modified_partial(double* const, const double* const, const double* const, uint64_t, double);
void loss_cross_entropy_partial(double* const, const double* const, const double* const, uint64_t, double);
void loss_hinge_partial(double* const, const double* const, const double* const, uint64_t, double);

void activation_sigmoid_partial(double* const, const double* const, uint64_t, double);
void activation_relu_partial(double* const, const double* const, uint64_t, double);
void activation_tanh_partial(double* const, const double* const, uint64_t, double);
void activation_linear_partial(double* const, const double* const, uint64_t, double);
void activation_relu_leaky_partial(double* const, const double* const, uint64_t, double);
void activation_relu_parametric_partial(double* const, const double* const, uint64_t, double);
void activation_elu_partial(double* const, const double* const, uint64_t, double);
void activation_softmax_partial(double* const, const double* const, uint64_t, double);
void activation_swish_partial(double* const, const double* const, uint64_t, double);
void activation_gelu_partial(double* const, const double* const, uint64_t, double);
void activation_selu_partial(double* const, const double* const, uint64_t, double);

#endif
