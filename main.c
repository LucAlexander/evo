#include "evo.h"
#include "kickstart.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <immintrin.h>

/* TODO
 * SIMD
 */

static activation_function activations[] = {
	activation_sigmoid,
	activation_relu,
	activation_tanh,
	activation_binary_step,
	activation_linear,
	activation_relu_leaky,
	activation_relu_parametric,
	activation_elu,
	activation_softmax,
	activation_swish,
	activation_gelu,
	activation_selu
};

static activation_function activation_partials[] = {
	activation_sigmoid_partial,
	activation_relu_partial,
	activation_tanh_partial,
	activation_linear_partial,
	activation_relu_leaky_partial,
	activation_relu_parametric_partial,
	activation_elu_partial,
	activation_softmax_partial,
	activation_swish_partial,
	activation_gelu_partial,
	activation_selu_partial
};

static loss_function losses[] = {
	loss_mse,
	loss_mae,
	loss_mape,
	loss_huber,
	loss_huber_modified,
	loss_cross_entropy,
	loss_hinge
};

static loss_derivative loss_partials[] = {
	loss_mse_partial,
	loss_mae_partial,
	loss_mape_partial,
	loss_huber_partial,
	loss_huber_modified_partial,
	loss_cross_entropy_partial,
	loss_hinge_partial
};

static bias_init bias_inits[] = {
	bias_initialization_zero,
	bias_initialization_const_flat,
	bias_initialization_const_uneven
};

static weight_init weight_inits[] = {
	weight_initialization_xavier,
	weight_initialization_he,
	weight_initialization_lecun,
	weight_initialization_uniform,
	weight_initialization_normal
};

network
network_init(
	pool* const mem,
	layer* const input, layer* const output,
	WEIGHT_FUNC weight, BIAS_FUNC bias,
	double weight_a, double weight_b,
	double bias_a, double bias_b,
	uint64_t batch_size, double learning_rate,
	LOSS_FUNC l
){
	network net = {
		.mem = mem,
		.temp = pool_alloc(TEMP_POOL_SIZE, POOL_STATIC),
		.nodes = pool_request(mem, sizeof(layer*)*2),
		.node_count = 0,
		.node_capacity = 2,
		.input = input,
		.output = output,
		.loss_output = pool_request(mem, sizeof(double)*output->data.layer.width),
		.loss = l,
		.derivative=l,
		.bias = bias,
		.weight = weight,
		.batch_size = batch_size,
		.learning_rate = learning_rate,
		.weight_parameter_a = weight_a,
		.weight_parameter_b = weight_b,
		.bias_parameter_a= bias_a,
		.bias_parameter_b= bias_b
	};
	return net;
}

uint64_t
network_register_layer(network* const net, layer* const node){
	if (net->node_count >= net->node_capacity){
		layer** new = pool_request(net->mem, sizeof(layer*)*net->node_capacity*2);
		for (uint64_t i = 0;i<net->node_capacity;++i){
			new[i] = net->nodes[i];
		}
		net->nodes = new;
		net->node_capacity *= 2;
	}
	net->nodes[net->node_count] = node;
	net->node_count += 1;
	return net->node_count - 1;
}

void
network_build(network* const net){
	sort_connections(net, NULL, net->input, net->input->pass_index+1);
	allocate_weights(net, net->mem, net->input, net->input->pass_index+1);
	reset_simulation_flags(net, net->input);
	init_params(net, net->input, net->input->pass_index+1);
}

layer*
input_init(pool* const mem, uint64_t width){
	layer* input = pool_request(mem, sizeof(layer));
	input->tag = INPUT_NODE;
	input->data.input.output = pool_request(mem, sizeof(double)*width);
	input->data.input.width = width;
	input->prev = NULL;
	input->prev_count = 0;
	input->prev_capacity = 0;
	input->next = pool_request(mem, 2*sizeof(uint64_t));
	input->next_count = 0;
	input->next_capacity = 2;
	input->simulated = 0;
	input->pass_index = 0;
	input->back_direction = 0;
	return input;
}

layer*
layer_init(pool* const mem, uint64_t width, ACTIVATION_FUNC activation, uint64_t parameter_a){
	layer* node = pool_request(mem, sizeof(layer));
	node->tag = LAYER_NODE;
	node->data.layer.output = pool_request(mem, sizeof(double)*width);
	node->data.layer.activated = pool_request(mem, sizeof(double)*width);
	node->data.layer.width = width;
	node->data.layer.weights = NULL;
	node->data.layer.bias = pool_request(mem, sizeof(double)*width);
	node->data.layer.bias_gradients = pool_request(mem, sizeof(double)*width);
	node->data.layer.activation_gradients = pool_request(mem, sizeof(double)*width);
	node->data.layer.activation = activation;
	node->data.layer.derivative = activation;
	node->data.layer.parameter_a = parameter_a;
	node->prev = pool_request(mem, 2*sizeof(uint64_t));
	node->prev_count = 0;
	node->prev_capacity = 2;
	node->next = pool_request(mem, 2*sizeof(uint64_t));
	node->next_count = 0;
	node->next_capacity = 2;
	node->simulated = 0;
	node->branched = 0;
	node->pass_index = 0;
	node->back_direction = 0;
	return node;
}

void
layer_link(network* const net, pool* const mem, uint64_t a, uint64_t b){
	assert(a < net->node_count);
	assert(b < net->node_count);
	layer* node_a = net->nodes[a];
	layer* node_b = net->nodes[b];
	assert(node_b->tag != INPUT_NODE);
	if (node_a->next_count == node_a->next_capacity){
		uint64_t* new = pool_request(mem, sizeof(uint64_t)*node_a->next_capacity*2);
		for (uint64_t i = 0;i<node_a->next_capacity;++i){
			new[i] = node_a->next[i];
		}
		node_a->next = new;
		node_a->next_capacity *= 2;
	}
	node_a->next[node_a->next_count] = b;
	node_a->next_count += 1;
	if (node_b->prev_count == node_b->prev_capacity){
		uint64_t* new = pool_request(mem, sizeof(uint64_t)*node_b->prev_capacity*2);
		for (uint64_t i = 0;i<node_b->prev_capacity;++i){
			new[i] = node_b->prev[i];
		}
		node_b->prev = new;
		node_b->prev_capacity *= 2;
	}
	node_b->prev[node_b->prev_count] = a;
	node_b->prev_count += 1;
}

void
layer_link_backward(network* const net, pool* const mem, uint64_t a, uint64_t b){
	assert(a < net->node_count);
	assert(b < net->node_count);
	layer* node_b = net->nodes[b];
	if (node_b->prev_count == node_b->prev_capacity){
		uint64_t* new = pool_request(mem, sizeof(uint64_t)*node_b->prev_capacity*2);
		for (uint64_t i = 0;i<node_b->prev_capacity;++i){
			new[i] = node_b->prev[i];
		}
		node_b->prev = new;
		node_b->prev_capacity *= 2;
	}
	node_b->prev[node_b->prev_count] = a;
	node_b->prev_count += 1;
}

void
layer_unlink(network* const net, uint64_t a, uint64_t b){
	assert(a < net->node_count);
	assert(b < net->node_count);
	layer* node_a = net->nodes[a];
	layer* node_b = net->nodes[b];
	assert(node_b->tag != INPUT_NODE);
	uint8_t found = 0;
	for (uint64_t i = 0;i<node_a->next_count;++i){
		if (node_a->next[i] == b){
			found = 1;
		}
		else if (found == 1){
			node_a->next[i-1] = node_a->next[i];
		}
	}
	assert(found == 1);
	node_a->next_count -= 1;
	found = 0;
	for (uint64_t i = 0;i<node_b->prev_count;++i){
		if (node_b->prev[i] == a){
			found = 1;
		}
		else if (found == 1){
			node_b->prev[i-1] = node_b->prev[i];
		}
	}
	assert(found == 1);
	node_b->prev_count -= 1;
}

void
layer_insert(network* const net, pool* const mem, uint64_t a, uint64_t b, uint64_t c){
	layer_unlink(net, a, c);
	layer_link(net, mem, a, b);
	layer_link(net, mem, b, c);
}

void
reset_simulation_flags(network* const net, layer* const node){
	if (node->simulated == 0){
		return;
	}
	node->simulated = 0;
	for (uint64_t i = 0;i<node->next_count;++i){
		reset_simulation_flags(net, net->nodes[node->next[i]]);
	}
}

void
sort_connections(network* const net, layer* const prev, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	for (uint64_t i = 0;i<node->next_count;++i){
		for (uint64_t k = i;k<node->next_count;++k){
			if (node->next[k] < node->next[i]){
				uint64_t temp = node->next[k];
				node->next[k] = node->next[i];
				node->next[i] = temp;
			}
		}
	}
	if (node->tag == LAYER_NODE){
		assert(prev != NULL);
		for (uint64_t i = 0;i<node->prev_count;++i){
			for (uint64_t k = i;k<node->prev_count;++k){
				if (node->prev[k] < node->prev[i]){
					uint64_t temp = node->prev[k];
					node->prev[k] = node->prev[i];
					node->prev[i] = temp;
				}
			}
			if (net->nodes[node->prev[i]] == prev){
				node->back_direction = i;
			}
		}
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		sort_connections(net, node, net->nodes[node->next[i]], pass_index);
	}
}

void
allocate_weights(network* const net, pool* const mem, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			allocate_weights(net, mem, net->nodes[node->next[i]], pass_index);
		}
		return;
	}
	node->data.layer.weights = pool_request(mem, sizeof(double*)*node->data.layer.width);
	node->data.layer.weight_gradients = pool_request(mem, sizeof(double)*node->data.layer.width);
	uint64_t sum = 0;
	for (uint64_t i = 0;i<node->prev_count;++i){
		layer* prev = net->nodes[node->prev[i]];
		sum += prev->data.layer.width;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.weights[i] = pool_request(mem, sizeof(double)*sum);
		node->data.layer.weight_gradients[i] = pool_request(mem, sizeof(double)*sum);
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		allocate_weights(net, mem, net->nodes[node->next[i]], pass_index);
	}
}

void
forward(network* const net, layer* const node, uint64_t pass_index){
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			forward(net, net->nodes[node->next[i]], pass_index);
		}
		return;
	}
	node->branched = 0;
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.output[i] = node->data.layer.bias[i];
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = net->nodes[node->prev[p]];
			if (prev->simulated == 0){
				weight_index += prev->data.layer.width;
				continue;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				double w = node->data.layer.weights[i][weight_index];
				node->data.layer.output[i] += prev->data.layer.activated[k] * w;
				weight_index += 1;
			}
		}
	}
	activations[node->data.layer.activation](
		node->data.layer.activated,
		node->data.layer.output,
		node->data.layer.width,
		node->data.layer.parameter_a
	);
	node->simulated = 1;
	for (uint64_t i = 0;i<node->next_count;++i){
		forward(net, net->nodes[node->next[i]], pass_index);
	}
}

void
backward(network* const net, layer* const node){
	if (node->tag == INPUT_NODE){
		return;
	}
	pool_empty(&net->temp);
	double* dadz = pool_request(&net->temp, sizeof(double)*node->data.layer.width);
	double* dcda = node->data.layer.activation_gradients;
	activation_partials[node->data.layer.derivative](
		dadz,
		node->data.layer.output,
		node->data.layer.width,
		node->data.layer.parameter_a
	);
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.bias_gradients[i] += 1 * dadz[i] * dcda[i];
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = net->nodes[node->prev[p]];
			if (prev->tag == INPUT_NODE){
				for (uint64_t k = 0;k<prev->data.layer.width;++k){
					double dzdw = prev->data.input.output[k];
					node->data.layer.weight_gradients[i][weight_index] += dzdw * dadz[i] * dcda[i];
					weight_index += 1;
				}
				continue;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				double dzdw = prev->data.layer.activated[k];
				node->data.layer.weight_gradients[i][weight_index] += dzdw * dadz[i] * dcda[i];
				weight_index += 1;
			}
		}
	}
	for (uint64_t i = 0;i<node->prev_count;++i){
		layer* prev = net->nodes[node->prev[i]];
		if (prev->tag == INPUT_NODE){
			continue;
		}
		for (uint64_t k = 0;k<prev->next_count;++k){
			if (net->nodes[prev->next[k]] != node){
				continue;
			}
			for (uint64_t t = 0;t<prev->data.layer.width;++t){
				for (uint64_t n = 0;n<node->data.layer.width;++n){
					double w = node->data.layer.weights[n][i];
					prev->data.layer.activation_gradients[t] += w * dadz[n] * dcda[n];
				}
			}
			break;
		}
	}
	if (node->prev_count > 1){
		if (node->branched == 1){
			backward(net, net->nodes[node->prev[node->back_direction]]);
			return;
		}
		node->branched = 1;
	}
	for (uint64_t i = 0;i<node->prev_count;++i){
		backward(net, net->nodes[node->prev[i]]);
	}
}

void
apply_gradients(network* const net, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			apply_gradients(net, net->nodes[node->next[i]], pass_index);
		}
		return;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		double average = (node->data.layer.bias_gradients[i]/net->batch_size);
		node->data.layer.bias[i] += net->learning_rate * average;
		node->data.layer.bias_gradients[i] = 0;
		node->data.layer.activation_gradients[i] = 0;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = net->nodes[node->prev[p]];
			if (prev->simulated == 0){
				weight_index += prev->data.layer.width;
				continue;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				double average = (node->data.layer.weight_gradients[i][weight_index]/net->batch_size);
				node->data.layer.weights[i][weight_index] += net->learning_rate * average;
				node->data.layer.weight_gradients[i][weight_index] = 0;
				weight_index += 1;
			}
		}
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		apply_gradients(net, net->nodes[node->next[i]], pass_index);
	}
}

void
zero_gradients(network* const net, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			zero_gradients(net, net->nodes[node->next[i]], pass_index);
		}
		return;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.bias_gradients[i] = 0;
		node->data.layer.activation_gradients[i] = 0;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = net->nodes[node->prev[p]];
			if (prev->simulated == 0){
				weight_index += prev->data.layer.width;
				continue;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				node->data.layer.weight_gradients[i][weight_index] = 0;
				weight_index += 1;
			}
		}
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		zero_gradients(net, net->nodes[node->next[i]], pass_index);
	}
}

void
network_train(network* const net, double** data, uint64_t data_size, double** expected){
	assert(data_size % net->batch_size == 0);
	uint64_t pass = net->input->pass_index+1;
	zero_gradients(net, net->input, pass);
	pass += 1;
	uint64_t i = 0;
	while (i<data_size){
		for (uint64_t k = i;i<k+net->batch_size;++i){
			memcpy(net->input->data.input.output, data[i], net->input->data.input.width*sizeof(double));
			forward(net, net->input, pass);
			pass += 1;
			double loss = losses[net->loss](
				net->loss_output,
				net->output->data.layer.activated,
				expected[i],
				net->output->data.layer.width,
				net->loss_parameter_a
			);
			printf("loss: %lf\n", loss);
			loss_partials[net->derivative](
				net->output->data.layer.activation_gradients,
				net->output->data.layer.activated,
				expected[i],
				net->output->data.layer.width,
				net->loss_parameter_a
			);
			backward(net, net->output);
		}
		apply_gradients(net, net->input, pass);
		pass += 1;
		reset_simulation_flags(net, net->input);
	}
}

void
init_params(network* const net, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			init_params(net, net->nodes[node->next[i]], pass_index);
		}
		return;
	}
	uint64_t sum = 0;
	for (uint64_t i = 0;i<node->prev_count;++i){
		sum += net->nodes[node->prev[i]]->data.layer.width;
	}
	weight_inits[net->weight](
		node->data.layer.weights,
		sum, node->data.layer.width,
		net->weight_parameter_a, net->weight_parameter_b
	);
	bias_inits[net->bias](
		node->data.layer.bias,
		node->data.layer.width,
		net->bias_parameter_a, net->bias_parameter_b
	);
	for (uint64_t i = 0;i<node->next_count;++i){
		init_params(net, net->nodes[node->next[i]], pass_index);
	}
}

void
set_seed(time_t seed){
	srandom(seed);
}

double
uniform_distribution(double min, double max){
	double n = ((double)random())/RAND_MAX;
	return (n*(max-min))+min;
}

double
normal_distribution(double mean, double std){
	double u1 = uniform_distribution(0, 1);
	double u2 = uniform_distribution(0, 1);
	double z0 = sqrtf(-2.8*logf(u1))*cos(2.0*M_PI*u2);
	return mean+std*z0;
}

void
bias_initialization_zero(double* const buffer, uint64_t size, double a, double b){
	memset(buffer, 0, size*sizeof(double));
}

void
bias_initialization_const_flat(double* const buffer, uint64_t size, double a, double b){
	memset(buffer, a, size*sizeof(double));
}

void
bias_initialization_const_uneven(double* const buffer, uint64_t size, double a, double b){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = normal_distribution(a, b);
	}
}

void
weight_initialization_xavier(double** const out, uint64_t in_size, uint64_t out_size, double aa, double b){
	float a = sqrtf(1/(in_size+out_size));
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++k){
			out[i][k] = uniform_distribution(-a, a);
		}
	}
}

void
weight_initialization_he(double** const out, uint64_t in_size, uint64_t out_size, double aa, double b){
	float a = sqrtf(6/in_size);
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++k){
			out[i][k] = uniform_distribution(-a, a);
		}
	}
}

void
weight_initialization_lecun(double** const out, uint64_t in_size, uint64_t out_size, double a, double b){
	float std = sqrtf(1/in_size);
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++k){
			out[i][k] = normal_distribution(0, std);
		}
	}
}

void
weight_initialization_uniform(double** const out, uint64_t in_size, uint64_t out_size, double a, double b){
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++k){
			out[i][k] = uniform_distribution(a, b);
		}
	}
}

void
weight_initialization_normal(double** const out, uint64_t in_size, uint64_t out_size, double a, double b){
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++k){
			out[i][k] = normal_distribution(a, b);
		}
	}
}

void
loss_mse_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = 2*(result[i]-expected[i]);
	}
}

void
loss_mae_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double term = result[i]-expected[i];
		output[i] = term/fabsf(term);
	}
}

void
loss_mape_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		output[i] = (1/powf(expected[i], 2))*term/fabsf(term);
	}
}

void
loss_huber_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		if (term <= a){
			output[i] = term;
			continue;
		}
		output[i] = a*(int)((0<term)-(term<0));
	}
}

void
loss_huber_modified_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	double coef = 1/a;
	for (uint64_t i = 0;i<size;++i){
		float term = result[i]-expected[i];
		if (expected[i]-result[i] <= a){
			output[i] = term*coef;
			continue;
		}
		output[i] = (int)((0<term)-(term-0));
	}
}

void
loss_cross_entropy_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = (-(expected[i]/result[i]))+((1-expected[i])/(1-result[i]));
	}
}

void
loss_hinge_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (expected[i]*result[i] >= 1){
			output[i] = 0;
		}
		output[i] = -expected[i];
	}
}

void
activation_sigmoid_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		float fx = 1/(1+expf(-buffer[i]));
		output[i] = fx*(1-fx);
	}
}

void
activation_relu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = (buffer[i] > 0);
	}
}

void
activation_tanh_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = 1-powf(tanh(buffer[i]), 2);
	}
}

void
activation_linear_partial(double* const output, const double* const buffer, uint64_t size, double a){
	memset(output, 1, size*sizeof(float));
}

void
activation_relu_leaky_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = 0.01;
	}
}

void
activation_relu_parametric_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a;
	}
}

void
activation_elu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a+(a*(expf(buffer[i])-1));
	}
}

void
activation_softmax_partial(double* const output, const double* const buffer, uint64_t size, double a){
	float sum = 0.0f;
	float* softmax_values = malloc(sizeof(float)*size);
	for (uint64_t i = 0;i<size;++i){
		softmax_values[i] = exp(buffer[i]);
		sum += softmax_values[i];
	}
	for (uint64_t i = 0;i<size;++i){
		softmax_values[i] /= sum;
	}
	for (uint64_t i = 0;i<size;++i){
		output[i] = softmax_values[i]*(1-softmax_values[i]);
		for (uint64_t j = 0;j<size;++j){
			if (i!=j){
				output[i] -= softmax_values[i]*softmax_values[j];
			}
		}
	}
	free(softmax_values);
}

void
activation_swish_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		float fx = 1/(1+expf(-buffer[i]*a));
		output[i] = fx+a*buffer[i]*fx*(1-fx);
	}
}

void
activation_gelu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	float s2op = sqrt(2/M_PI);
	for (uint64_t i = 0;i<size;++i){
		float x = buffer[i];
		float inside = tanh(s2op*(x+GELU_C*powf(x, 3)));
		output[i] = 0.5*(1+inside)+0.5*x*(1-powf(inside, 2))*s2op*(1+3*GELU_C*x*x);
	}
}

void
activation_selu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			output[i] = SELU_LAMBDA;
			continue;
		}
		output[i] = SELU_LAMBDA*SELU_ALPHA*expf(x);
	}
}

double
loss_mse(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += loss*loss;
	}
	return (sum)/(size);
}

double
loss_mae(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += abs(loss);
	}
	return sum/(size);
}

double
loss_mape(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float loss = expect-result[i];
		buffer[i] = loss;
		sum += abs(loss/expect);
	}
	return sum/(size);
}

double
loss_huber(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	float hpsq = a*a*0.5;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		float x = expect-res;
		buffer[i] = x;
		if (abs(x) <= a){
			sum += x*x*0.5;
			continue;
		}
		sum += (a*abs(x))-hpsq;
	}
	return sum;
}

double
loss_huber_modified(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		float x = expect-res;
		buffer[i] = x;
		x = expect*res;
		if (x > -1){
			sum += pow(fmax(0, 1-x),2);
			continue;
		}
		sum -= 4*x;
	}
	return sum;
}

double
loss_cross_entropy(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += expect*log(res);
	}
	return -sum;
}

double
loss_hinge(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += fmax(0,1-(expect*res));
	}
	return sum;
}

void
activation_sigmoid(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = 1/(1+expf(-output[i]));
	}
}

void
activation_relu(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = fmax(0,output[i]);
	}
}

void
activation_tanh(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = tanh(output[i]);
	}
}

void
activation_binary_step(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = output[i] >= 0;
	}
}

void
activation_linear(double* const buffer, const double* const output, uint64_t size, double a){
	memcpy(buffer, output, size*sizeof(double));
}

void
activation_relu_leaky(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = fmax(0.1*x,x);
	}
}

void
activation_relu_parametric(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = fmax(a*x, x);
	}
}

void
activation_elu(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		if (x < 0){
			buffer[i] = a*(expf(x)-1);
		}
	}
}

void
activation_softmax(double* const buffer, const double* const output, uint64_t size, double a){
	float denom = 0;
	for (uint64_t i = 0;i<size;++i){
		denom += expf(output[i]);
	}
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = expf(output[i])/denom;
	}
}

void
activation_swish(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = x/(1+expf(-x));
	}
}

void
activation_gelu(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	const float s2p  = sqrtf(2/M_PI);
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = 0.5*x*(1+tanh(s2p*(x+(GELU_C*pow(x,3)))));
	}
}

void
activation_selu(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			buffer[i] = SELU_LAMBDA * x;
			continue;
		}
		buffer[i] = SELU_LAMBDA*SELU_ALPHA*(expf(x)-1);
	}
}

void
write_node(network* const net, layer* const node, FILE* outfile){
	fwrite(&node->tag, sizeof(uint8_t), 1, outfile);
	fwrite(&node->data.layer.width, sizeof(uint64_t), 1, outfile);
	if (node->tag == LAYER_NODE){
		uint64_t sum = 0;
		for (uint64_t i = 0;i<node->prev_count;++i){
			sum += net->nodes[node->prev[i]]->data.layer.width;
		}
		fwrite(&sum, sizeof(uint64_t), 1, outfile);
		for (uint64_t i = 0;i<node->data.layer.width;++i){
			fwrite(&node->data.layer.weights[i], sizeof(double), sum, outfile);
		}
		fwrite(&node->data.layer.bias, sizeof(double), node->data.layer.width, outfile);
		fwrite(&node->data.layer.activation, sizeof(ACTIVATION_FUNC), 1, outfile);
		fwrite(&node->data.layer.derivative, sizeof(ACTIVATION_PARTIAL_FUNC), 1, outfile);
		fwrite(&node->data.layer.parameter_a, sizeof(uint64_t), 1, outfile);
	}
	fwrite(&node->next_count, sizeof(uint64_t), 1, outfile);
	fwrite(&node->next, sizeof(uint64_t), node->next_count, outfile);
}

void
load_nodes(network* const net, pool* const mem, FILE* infile){
	layer* node = pool_request(mem, sizeof(layer));
	uint64_t err = fread(&node->tag, sizeof(uint8_t), 1, infile);
	while (err != 0){
		fread(&node->data.layer.width, sizeof(uint64_t), 1, infile);
		node->data.layer.output = pool_request(mem, sizeof(double)*node->data.layer.width);
		if (node->tag == LAYER_NODE){
			uint64_t sum;
			fread(&sum, sizeof(uint64_t), 1, infile);
			node->data.layer.weights = pool_request(mem, sizeof(double*)*node->data.layer.width);
			node->data.layer.weight_gradients = pool_request(mem, sizeof(double*)*node->data.layer.width);
			for (uint64_t i = 0;i<node->data.layer.width;++i){
				node->data.layer.weights[i] = pool_request(mem, sizeof(double)*sum);
				node->data.layer.weight_gradients[i] = pool_request(mem, sizeof(double)*sum);
				fread(&node->data.layer.weights[i], sizeof(double), sum, infile);
			}
			node->data.layer.bias = pool_request(mem, sizeof(double)*node->data.layer.width);
			fread(&node->data.layer.bias, sizeof(double), node->data.layer.width, infile);
			fread(&node->data.layer.activation, sizeof(ACTIVATION_FUNC), 1, infile);
			fread(&node->data.layer.derivative, sizeof(ACTIVATION_PARTIAL_FUNC), 1, infile);
			fread(&node->data.layer.parameter_a, sizeof(uint64_t), 1, infile);
			node->data.layer.activated = pool_request(mem, sizeof(double)*node->data.layer.width);
			node->data.layer.bias_gradients = pool_request(mem, sizeof(double)*node->data.layer.width);
			node->data.layer.activation_gradients = pool_request(mem, sizeof(double)*node->data.layer.width);
		}
		else {
			net->input = node;
		}
		fread(&node->next_count, sizeof(uint64_t), 1, infile);
		node->next_capacity = node->next_count;
		node->next = pool_request(mem, sizeof(uint64_t)*node->next_capacity);
		fread(&node->next, sizeof(uint64_t), node->next_count, infile);
		if (node->next_count == 0){
			net->output = node;
		}
		node->pass_index = 0;
		node->simulated = 0;
		node->prev_count = 0;
		node->prev_capacity = 2;
		node->prev = pool_request(mem, sizeof(uint64_t)*2);
		network_register_layer(net, node);
		node = pool_request(mem, sizeof(layer));
		err = fread(&node->tag, sizeof(uint8_t), 1, infile);
	}
}

void
write_network(network* const net, const char* filename){
	FILE* outfile = fopen(filename, "wb");
	assert(outfile != NULL);
	fwrite(&net->loss, sizeof(LOSS_FUNC), 1, outfile);
	fwrite(&net->derivative, sizeof(LOSS_PARTIAL_FUNC), 1, outfile);
	fwrite(&net->bias, sizeof(BIAS_FUNC), 1, outfile);
	fwrite(&net->weight, sizeof(WEIGHT_FUNC), 1, outfile);
	fwrite(&net->batch_size, sizeof(uint64_t), 1, outfile);
	fwrite(&net->learning_rate, sizeof(double), 1, outfile);
	fwrite(&net->loss_parameter_a, sizeof(double), 1, outfile);
	fwrite(&net->weight_parameter_a, sizeof(double), 1, outfile);
	fwrite(&net->weight_parameter_b, sizeof(double), 1, outfile);
	fwrite(&net->bias_parameter_a, sizeof(double), 1, outfile);
	fwrite(&net->bias_parameter_b, sizeof(double), 1, outfile);
	fwrite(&net->node_capacity, sizeof(uint64_t), 1, outfile);
	for (uint64_t i = 0;i<net->node_count;++i){
		write_node(net, net->nodes[i], outfile);
	}
	fclose(outfile);
}

network
load_network(pool* const mem, const char* filename){
	FILE* infile = fopen(filename, "rb");
	assert(infile != NULL);
	network net = {
		.mem = mem,
		.temp = pool_alloc(TEMP_POOL_SIZE, POOL_STATIC)
	};
	fread(&net.loss, sizeof(LOSS_FUNC), 1, infile);
	fread(&net.derivative, sizeof(LOSS_PARTIAL_FUNC), 1, infile);
	fread(&net.bias, sizeof(BIAS_FUNC), 1, infile);
	fread(&net.weight, sizeof(WEIGHT_FUNC), 1, infile);
	fread(&net.batch_size, sizeof(uint64_t), 1, infile);
	fread(&net.learning_rate, sizeof(double), 1, infile);
	fread(&net.loss_parameter_a, sizeof(double), 1, infile);
	fread(&net.weight_parameter_a, sizeof(double), 1, infile);
	fread(&net.weight_parameter_b, sizeof(double), 1, infile);
	fread(&net.bias_parameter_a, sizeof(double), 1, infile);
	fread(&net.bias_parameter_b, sizeof(double), 1, infile);
	fread(&net.node_capacity, sizeof(uint64_t), 1, infile);
	net.nodes = pool_request(mem, sizeof(uint64_t)*net.node_capacity);
	load_nodes(&net, mem, infile);
	fclose(infile);
	net.loss_output = pool_request(mem, sizeof(double)*net.input->data.input.width);
	for (uint64_t i = 0;i<net.node_count;++i){
		layer* node = net.nodes[i];
		for (uint64_t k = 0;k<node->next_count;++k){
			layer_link_backward(&net, mem, i, node->next[k]);
		}
	}
	sort_connections(&net, NULL, net.input, net.input->pass_index+1);
	net.input->pass_index += 1;
	return net;
}

prediction
predict(network* const net, double* input, uint64_t len){
	assert(len == net->input->data.input.width);
	memcpy(net->input->data.input.output, input, len*sizeof(double));
	forward(net, net->input, net->input->pass_index+1);
	net->input->pass_index += 1;
	prediction pred = {
		.class = 0,
		.probability = 0
	};
	for (uint64_t i = 0;i<net->output->data.layer.width;++i){
		if (net->output->data.layer.activated[i] > pred.probability){
			pred.class = i;
			pred.probability = net->output->data.layer.activated[i];
		}
	}
	return pred;
}


prediction_vector
predict_vector(network* const net, pool* const mem, double** input, uint64_t vector_len, uint64_t len){
	assert(len == net->input->data.input.width);
	assert(vector_len == net->batch_size);
	prediction_vector pred = {
		.class = pool_request(mem, sizeof(uint64_t)*vector_len),
		.probability= pool_request(mem, sizeof(double)*vector_len),
		.len = vector_len
	};
	for (uint64_t v = 0;v<vector_len;++v){
		memcpy(net->input->data.input.output, input[v], len*sizeof(double));
		forward(net, net->input, net->input->pass_index+1);
		net->input->pass_index += 1;
		uint64_t class = 0;
		double prob = 0;
		for (uint64_t i = 0;i<net->output->data.layer.width;++i){
			if (net->output->data.layer.activated[i] > prob){
				class = i;
				prob = net->output->data.layer.activated[i];
			}
		}
		pred.class[v] = class;
		pred.probability[v] = prob;
	}
	return pred;
}

prediction_vector
predict_vector_batched(network* const net, pool* const mem, double*** input, uint64_t sample_count, uint64_t vector_len, uint64_t len){
	prediction_vector pred = {
		.class = pool_request(mem, sizeof(uint64_t)*sample_count),
		.probability= pool_request(mem, sizeof(double)*sample_count),
		.len = sample_count 
	};
	pool_save(mem);
	for (uint64_t i = 0;i<sample_count;++i){
		prediction_vector sample = predict_vector(net, mem, input[i], vector_len, len);
		pred.class[i] = sample.class[vector_len-1];
		pred.probability[i] = sample.probability[vector_len-1];
		pool_load(mem);
	}
	return pred;
}

void
network_show(network* const net){
	for (uint64_t i = 0;i<net->node_count;++i){
		printf("layer node %lu:\n", i);
		layer* node = net->nodes[i];
		if (node->tag == INPUT_NODE){
			printf("(input)\n");
		}
		else if (node->next_count == 0){
			printf("(output)\n");
		}
		printf("neurons: %lu\n", node->data.layer.width);
		for (uint64_t k = 0;k<node->next_count;++k){
			printf(" * -> %lu\n", node->next[k]);
		}
		for (uint64_t k = 0;k<node->prev_count;++k){
			printf(" %lu -> * \n", node->prev[k]);
		}
		printf("\n");
	}
}

int
main(int argc, char** argv){
	set_seed(time(NULL));
	pool mem = pool_alloc(TEMP_POOL_SIZE, POOL_STATIC);


	layer* input = input_init(&mem, 32);
	layer* a = layer_init(&mem, 64, ACTIVATION_SIGMOID, 0);
	layer* b = layer_init(&mem, 64, ACTIVATION_SIGMOID, 0);
	layer* c = layer_init(&mem, 64, ACTIVATION_SIGMOID, 0);
	layer* output = layer_init(&mem, 2, ACTIVATION_SIGMOID, 0);


	network net = network_init(
		&mem,
		input, output,
		WEIGHT_INITIALIZATION_NORMAL,
		BIAS_INITIALIZATION_ZERO,
		1, 2,
		0, 0,
		32, 0.000001,
		LOSS_MSE
	);


	uint64_t input_id = network_register_layer(&net, input);
	uint64_t aid = network_register_layer(&net, a);
	uint64_t bid = network_register_layer(&net, b);
	uint64_t cid = network_register_layer(&net, c);
	uint64_t output_id = network_register_layer(&net, output);
	layer_link(&net, &mem, input_id, output_id);
	layer_insert(&net, &mem, input_id, cid, output_id);
	layer_insert(&net, &mem, input_id, bid, cid);
	layer_insert(&net, &mem, input_id, aid, bid);
	layer_link(&net, &mem, cid, aid);
	layer_link(&net, &mem, bid, output_id);


	network_build(&net);
	network_show(&net);

	write_network(&net, "test.mdl");
	printf("wrote\n");
	network copy = load_network(&mem, "test.mdl");
	printf("loaded\n");
	network_show(&copy);

	double** training_data = pool_request(&mem, sizeof(double*)*128);
	for (uint32_t i = 0;i<128;++i){
		training_data[i] = pool_request(&mem, sizeof(double)*net.input->data.input.width);
		for (uint32_t k = 0;k<net.input->data.input.width;++k){
			training_data[i][k] = random();
		}
	}
	double** expected = pool_request(&mem, sizeof(double*)*128);
	for (uint32_t i = 0;i<128;++i){
		expected[i] = pool_request(&mem, sizeof(double)*net.input->data.input.width);
		for (uint32_t k = 0;k<net.input->data.input.width;++k){
			expected[i][k] = random();
		}
	}

	network_train(&net, training_data, 128, expected);

	prediction_vector vec = predict_vector_batched(&net, &mem, &training_data, 1, 32, net.input->data.input.width);
	printf("predicted %lu (%lf) \n", vec.class[0], vec.probability[0]);

	pool_dealloc(&mem);
	return 0;
}
