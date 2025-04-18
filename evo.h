#ifndef EVOMORPH_H
#define EVOMORPH_H

#include "kickstart.h"
#include <time.h>
#include <stdio.h>

#ifdef __SSE__
#include <xmmintrin.h>

__m128d exp_neg_pd(__m128d x);
__m128d tanh_pd(__m128d x);
#endif

#define SSE_ALIGNMENT 16
#define DOUBLE_DIV 128/sizeof(double)

#define TEMP_POOL_SIZE 0x1000000
#define GELU_C 0.044715
#define SELU_LAMBDA 1.0507009873554804934193349852946
#define SELU_ALPHA 1.6732632423543772848170429916717

typedef void (*activation_function)(double* const, const double* const, uint64_t, double);
typedef double (*loss_function)(double* const, const double* const, const double* const, uint64_t, double);
typedef void (*loss_derivative)(double* const, const double* const, const double* const, uint64_t, double);
typedef void (*bias_init)(double* const, uint64_t, double, double);
typedef void (*weight_init)(double** const, uint64_t, uint64_t, double, double);
typedef void (*layer_weight_init)(double* const, uint64_t, double, double);

typedef enum LOSS_FUNC {
	LOSS_MSE,
	LOSS_MAE,
	LOSS_MAPE,
	LOSS_HUBER,
	LOSS_HUBER_MODIFIED,
	LOSS_CROSS_ENTROPY,
	LOSS_HINGE
} LOSS_FUNC;

typedef enum ACTIVATION_FUNC {
	ACTIVATION_SIGMOID,
	ACTIVATION_RELU,
	ACTIVATION_TANH,
	ACTIVATION_BINARY_STEP,
	ACTIVATION_LINEAR,
	ACTIVATION_RELU_LEAKY,
	ACTIVATION_RELU_PARAMETRIC,
	ACTIVATION_ELU,
	ACTIVATION_SOFTMAX,
	ACTIVATION_SWISH,
	ACTIVATION_GELU,
	ACTIVATION_SELU
} ACTIVATION_FUNC;

#define ACTIVATION_COUNT ACTIVATION_SELU+1

typedef enum BIAS_FUNC {
	BIAS_INITIALIZATION_ZERO,
	BIAS_INITIALIZATION_CONST_FLAT,
	BIAS_INITIALIZATION_CONST_UNEVEN,
} BIAS_FUNC;

typedef enum WEIGHT_FUNC {
	WEIGHT_INITIALIZATION_XAVIER,
	WEIGHT_INITIALIZATION_HE,
	WEIGHT_INITIALIZATION_LECUN,
	WEIGHT_INITIALIZATION_UNIFORM,
	WEIGHT_INITIALIZATION_NORMAL,
} WEIGHT_FUNC;

typedef enum LAYER_WEIGHT_FUNC {
	LAYER_WEIGHT_INITIALIZATION_UNIFORM,
	LAYER_WEIGHT_INITIALIZATION_NORMAL,
	LAYER_WEIGHT_INITIALIZATION_STRONG,
	LAYER_WEIGHT_INITIALIZATION_PARAMETRIC
} LAYER_WEIGHT_FUNC;

typedef enum LOSS_PARTIAL_FUNC {
	LOSS_MSE_PARTIAL,
	LOSS_MAE_PARTIAL,
	LOSS_MAPE_PARTIAL,
	LOSS_HUBER_PARTIAL,
	LOSS_HUBER_MODIFIED_PARTIAL,
	LOSS_CROSS_ENTROPY_PARTIAL,
	LOSS_HINGE_PARTIAL,
} LOSS_PARTIAL_FUNC;

typedef enum ACTIVATION_PARTIAL_FUNC {
	ACTIVATION_SIGMOID_PARTIAL,
	ACTIVATION_RELU_PARTIAL,
	ACTIVATION_TANH_PARTIAL,
	ACTIVATION_LINEAR_PARTIAL,
	ACTIVATION_RELU_LEAKY_PARTIAL,
	ACTIVATION_RELU_PARAMETRIC_PARTIAL,
	ACTIVATION_ELU_PARTIAL,
	ACTIVATION_SOFTMAX_PARTIAL,
	ACTIVATION_SWISH_PARTIAL,
	ACTIVATION_GELU_PARTIAL,
	ACTIVATION_SELU_PARTIAL
} ACTIVATION_PARTIAL_FUNC;

typedef struct layer layer;
typedef struct layer {
	uint64_t* prev;
	uint64_t prev_count;
	uint64_t prev_capacity;
	uint64_t* next;
	uint64_t next_count;
	uint64_t next_capacity;
	uint64_t pass_index;
	uint64_t back_direction;
	union {
		struct {
			uint64_t width;
			double* output;
			double* activated;
			// width * number of input weights, in same order as layer** prev
			double** weights;
			double* bias;
			double** weight_gradients;
			double* bias_gradients;
			double* activation_gradients;
			double* prev_weights;
			double* prev_weight_gradients;
			double parameter_a;
			uint64_t gradient_count;
			ACTIVATION_FUNC activation;
			ACTIVATION_PARTIAL_FUNC derivative;
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
	uint8_t branched;
} layer;

typedef struct network {
	pool* mem;
	pool temp;
	layer** nodes;
	uint64_t node_count;
	uint64_t node_capacity;
	layer* input;
	layer* output;
	LOSS_FUNC loss;
	LOSS_PARTIAL_FUNC derivative;
	double* loss_output;
	BIAS_FUNC bias;
	WEIGHT_FUNC weight;
	LAYER_WEIGHT_FUNC layer_weight;
	ACTIVATION_FUNC prune;
	uint64_t batch_size;
	double learning_rate;
	double loss_parameter_a;
	double weight_parameter_a;
	double weight_parameter_b;
	double bias_parameter_a;
	double bias_parameter_b;
	double prev_parameter_a;
	double prev_parameter_b;
	double prune_parameter_a;
	double gradient_clamp;
	uint8_t layers_weighted;
} network;

network network_init(pool* const mem, layer* const input, layer* const output, WEIGHT_FUNC w, BIAS_FUNC b, LAYER_WEIGHT_FUNC lw, ACTIVATION_FUNC prune, double weight_a, double weight_b, double bias_a, double bias_b, double prev_a, double prev_b, double prune_a, uint64_t batch_size, double learning_rate, double clamp, LOSS_FUNC l);
uint64_t network_register_layer(network* const net, layer* const node);
void reset_pass_index(network* const net);
void network_build(network* const net);
layer* input_init(pool* const mem, uint64_t width);
layer* layer_init(pool* const mem, uint64_t width, ACTIVATION_FUNC activation, double parameter_a);
void layer_link(network* const net, pool* const mem, uint64_t a, uint64_t b);
void layer_link_backward(network* const net, pool* const mem, uint64_t a, uint64_t b);
void layer_unlink(network* const net, uint64_t a, uint64_t b);
void layer_insert(network* const net, pool* const mem, uint64_t a, uint64_t b, uint64_t c);
void reset_simulation_flags(network* const net, layer* const node);
void sort_connections(network* const net, layer* const prev, layer* const node, uint64_t pass_index);
void allocate_node_weights(network* const net, pool* const mem, layer* const node);
void allocate_weights(network* const net, pool* const mem, layer* const node, uint64_t pass_index);
void clamp_gradient(network* const net, double* item);
void clamp_gradients(network* const net, double* const vector, uint64_t size);
void forward(network* const net, layer* const node, uint64_t pass_index);
void backward(network* const net, layer* const node);
void apply_gradients(network* const net, layer* const node, uint64_t pass_index);
void zero_gradients(network* const net, layer* const node, uint64_t pass_index);
void clear_activation_gradients(network* const net, layer* const node, uint64_t pass_index);
void network_train(network* const net, double** data, uint64_t data_size, double** expected);
void init_params(network* const net, layer* const node, uint64_t pass_index);

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

void layer_weight_initialization_uniform(double* const, uint64_t, double, double);
void layer_weight_initialization_normal(double* const, uint64_t, double, double);
void layer_weight_initialization_strong(double* const, uint64_t, double, double);
void layer_weight_initialization_parametric(double* const, uint64_t, double, double);

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

void write_node(network* const net, layer* const node, FILE* outfile);
void write_network(network* const net, const char* filename);
void load_nodes(network* const net, pool* const node, FILE* infile);
network load_network(pool* const mem, const char* filename);

typedef struct prediction {
	uint64_t class;
	double probability;
} prediction;

typedef struct prediction_vector {
	uint64_t* class;
	double* probability;
	uint64_t len;
} prediction_vector;

prediction predict(network* const net, double* input, uint64_t len);
prediction_vector predict_vector(network* const net, pool* const mem, double** input, uint64_t vector_len, uint64_t len);
prediction_vector predict_vector_batched(network* const net, pool* const mem, double*** input, uint64_t sample_count, uint64_t vector_len, uint64_t len);

void network_show(network* const net);

void network_prune(network* const net);
void network_compose_layer(network* const net, layer* const node);
void update_layer_connection_data(network* const net, layer* const node, uint64_t target_id);
void grow_network(network* const net, double** training_data, uint64_t samples, double** expected, uint64_t epochs, uint64_t prune_epoch, uint64_t grow_epoch);
void grow_network_retrain(network* const net, double** training_data, uint64_t samples, double** expected, uint64_t epochs, uint64_t prune_epoch, uint64_t grow_epoch);
layer* grow_layer(pool* const mem);

#endif
