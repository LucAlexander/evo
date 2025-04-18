#include "evo.h"
#include "kickstart.h"
#include <assert.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

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

static layer_weight_init layer_weight_inits[] = {
	layer_weight_initialization_uniform,
	layer_weight_initialization_normal,
	layer_weight_initialization_strong,
	layer_weight_initialization_parametric
};

network
network_init(
	pool* const mem,
	layer* const input, layer* const output,
	WEIGHT_FUNC weight, BIAS_FUNC bias,
	LAYER_WEIGHT_FUNC layer_weight, ACTIVATION_FUNC prune,
	double weight_a, double weight_b,
	double bias_a, double bias_b,
	double layer_a, double layer_b,
	double prune_a,
	uint64_t batch_size, double learning_rate,
	double clamp,
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
#ifdef __SSE__
		.loss_output = pool_request_aligned(mem, sizeof(double)*output->data.layer.width, SSE_ALIGNMENT),
#else
		.loss_output = pool_request(mem, sizeof(double)*output->data.layer.width),
#endif
		.loss = l,
		.derivative=l,
		.bias = bias,
		.weight = weight,
		.layer_weight = layer_weight,
		.prune = prune,
		.batch_size = batch_size,
		.learning_rate = learning_rate,
		.weight_parameter_a = weight_a,
		.weight_parameter_b = weight_b,
		.bias_parameter_a= bias_a,
		.bias_parameter_b= bias_b,
		.prev_parameter_a = layer_a,
		.prev_parameter_b = layer_b,
		.prune_parameter_a = prune_a,
		.gradient_clamp = clamp,
		.layers_weighted = 0
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
reset_pass_index(network* const net){
	for (uint64_t i = 0;i<net->node_count;++i){
		net->nodes[i]->pass_index = 0;
	}
}

void
network_build(network* const net){
	reset_pass_index(net);
	sort_connections(net, NULL, net->input, net->input->pass_index+1);
	allocate_weights(net, net->mem, net->input, net->input->pass_index+1);
	reset_simulation_flags(net, net->input);
	init_params(net, net->input, net->input->pass_index+1);
}

layer*
input_init(pool* const mem, uint64_t width){
	layer* input = pool_request(mem, sizeof(layer));
	input->tag = INPUT_NODE;
#ifdef __SSE__
	input->data.input.output = pool_request_aligned(mem, sizeof(double)*width, SSE_ALIGNMENT);
#else
	input->data.input.output = pool_request(mem, sizeof(double)*width);
#endif
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
layer_init(pool* const mem, uint64_t width, ACTIVATION_FUNC activation, double parameter_a){
	layer* node = pool_request(mem, sizeof(layer));
	node->tag = LAYER_NODE;
	node->data.layer.width = width;
	node->data.layer.weights = NULL;
	node->data.layer.prev_weights = NULL;
	node->data.layer.prev_weight_gradients = NULL;
#ifdef __SSE__
	node->data.layer.output = pool_request_aligned(mem, sizeof(double)*width, SSE_ALIGNMENT);
	node->data.layer.activated = pool_request_aligned(mem, sizeof(double)*width, SSE_ALIGNMENT);
	node->data.layer.bias = pool_request_aligned(mem, sizeof(double)*width, SSE_ALIGNMENT);
	node->data.layer.bias_gradients = pool_request_aligned(mem, sizeof(double)*width, SSE_ALIGNMENT);
	node->data.layer.activation_gradients = pool_request_aligned(mem, sizeof(double)*width, SSE_ALIGNMENT);
#else
	node->data.layer.output = pool_request(mem, sizeof(double)*width);
	node->data.layer.activated = pool_request(mem, sizeof(double)*width);
	node->data.layer.bias = pool_request(mem, sizeof(double)*width);
	node->data.layer.bias_gradients = pool_request(mem, sizeof(double)*width);
	node->data.layer.activation_gradients = pool_request(mem, sizeof(double)*width);
#endif
	node->data.layer.activation = activation;
	node->data.layer.derivative = activation;
	node->data.layer.parameter_a = parameter_a;
	node->data.layer.gradient_count = 0;
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
	for (uint64_t i = 0;i<node_a->next_count;++i){
		if (node_a->next[i] == b){
			return;
		}
	}
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
allocate_node_weights(network* const net, pool* const mem, layer* const node){
#ifdef __SSE__
	node->data.layer.weights = pool_request_aligned(mem, sizeof(double*)*node->data.layer.width, SSE_ALIGNMENT);
	node->data.layer.weight_gradients = pool_request_aligned(mem, sizeof(double)*node->data.layer.width, SSE_ALIGNMENT);
	node->data.layer.prev_weights = pool_request_aligned(mem, sizeof(double)*node->prev_count, SSE_ALIGNMENT);
	node->data.layer.prev_weight_gradients = pool_request_aligned(mem, sizeof(double)*node->prev_count, SSE_ALIGNMENT);
#else
	node->data.layer.weights = pool_request(mem, sizeof(double*)*node->data.layer.width);
	node->data.layer.weight_gradients = pool_request(mem, sizeof(double)*node->data.layer.width);
	node->data.layer.prev_weights = pool_request(mem, sizeof(double)*node->prev_count);
	node->data.layer.prev_weight_gradients = pool_request(mem, sizeof(double)*node->prev_count);
#endif
	uint64_t sum = 0;
	for (uint64_t i = 0;i<node->prev_count;++i){
		layer* prev = net->nodes[node->prev[i]];
		sum += prev->data.layer.width;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
#ifdef __SSE__
		node->data.layer.weights[i] = pool_request_aligned(mem, sizeof(double)*sum, SSE_ALIGNMENT);
		node->data.layer.weight_gradients[i] = pool_request_aligned(mem, sizeof(double)*sum, SSE_ALIGNMENT);
#else
		node->data.layer.weights[i] = pool_request(mem, sizeof(double)*sum);
		node->data.layer.weight_gradients[i] = pool_request(mem, sizeof(double)*sum);
#endif
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
	allocate_node_weights(net, mem, node);
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
	if (net->layers_weighted == 1){
		for (uint64_t i = 0;i<node->data.layer.width;++i){
			node->data.layer.output[i] = node->data.layer.bias[i];
			uint64_t weight_index = 0;
			for (uint64_t p = 0;p<node->prev_count;++p){
				layer* prev = net->nodes[node->prev[p]];
				if (prev->simulated == 0){
					weight_index += prev->data.layer.width;
					continue;
				}
				double layer_weight = node->data.layer.prev_weights[p];
				for (uint64_t k = 0;k<prev->data.layer.width;++k){
					double w = node->data.layer.weights[i][weight_index];
					node->data.layer.output[i] += prev->data.layer.activated[k] * w * layer_weight;
					weight_index += 1;
				}
			}
		}
	}
	else{
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
clamp_gradients(network* const net, double* const vector, uint64_t size){
	for (uint64_t i = 0;i<size;++i){
		if (vector[i] > net->gradient_clamp){
			vector[i] = net->gradient_clamp;
		}
		if (vector[i] < -net->gradient_clamp){
			vector[i] = -net->gradient_clamp;
		}
	}
}

void
clamp_gradient(network* const net, double* item){
	if (*item > net->gradient_clamp){
		*item = net->gradient_clamp;
	}
	if (*item < -net->gradient_clamp){
		*item = -net->gradient_clamp;
	}
}

void
backward(network* const net, layer* const node){
	if (node->tag == INPUT_NODE){
		return;
	}
	pool_empty(&net->temp);
#ifdef __SSE__
	double* dadz = pool_request_aligned(&net->temp, sizeof(double)*node->data.layer.width, SSE_ALIGNMENT);
#else
	double* dadz = pool_request(&net->temp, sizeof(double)*node->data.layer.width);
#endif
	double* dcda = node->data.layer.activation_gradients;
	node->data.layer.gradient_count += 1;
	activation_partials[node->data.layer.derivative](
		dadz,
		node->data.layer.output,
		node->data.layer.width,
		node->data.layer.parameter_a
	);
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.bias_gradients[i] += 1 * dadz[i] * dcda[i];
	}
	clamp_gradients(net, node->data.layer.bias_gradients, node->data.layer.width);
	if (net->layers_weighted == 1){
		for (uint64_t i = 0;i<node->data.layer.width;++i){
			uint64_t weight_index = 0;
			for (uint64_t p = 0;p<node->prev_count;++p){
				layer* prev = net->nodes[node->prev[p]];
				double layer_weight = node->data.layer.prev_weights[p];
				if (prev->tag == INPUT_NODE){
					for (uint64_t k = 0;k<prev->data.layer.width;++k){
						double dzdw = prev->data.input.output[k] * layer_weight;
						node->data.layer.weight_gradients[i][weight_index] += dzdw * dadz[i] * dcda[i];
						clamp_gradient(net, &node->data.layer.weight_gradients[i][weight_index]);
						weight_index += 1;
					}
					continue;
				}
				for (uint64_t k = 0;k<prev->data.layer.width;++k){
					double dzdw = prev->data.layer.activated[k] * layer_weight;
					node->data.layer.weight_gradients[i][weight_index] += dzdw * dadz[i] * dcda[i];
					clamp_gradient(net, &node->data.layer.weight_gradients[i][weight_index]);
					weight_index += 1;
				}
			}
		}
		uint64_t weight_index = 0;
		for (uint64_t i = 0;i<node->prev_count;++i){
			layer* prev = net->nodes[node->prev[i]];
			if (prev->tag == INPUT_NODE){
				weight_index += prev->data.input.width;
				continue;
			}
			double layer_weight = node->data.layer.prev_weights[i];
			for (uint64_t k = 0;k<prev->next_count;++k){
				if (net->nodes[prev->next[k]] != node){
					continue;
				}
				for (uint64_t t = 0;t<prev->data.layer.width;++t){
					for (uint64_t n = 0;n<node->data.layer.width;++n){
						double w = node->data.layer.weights[n][weight_index] * layer_weight;
						prev->data.layer.activation_gradients[t] += w * dadz[n] * dcda[n];
					}
					weight_index += 1;
				}
				clamp_gradients(net, prev->data.layer.activation_gradients, prev->data.layer.width);
				break;
			}
		}
		double sig_dadz = 0;
		double sig_dcda = 0;
		for (uint64_t i = 0;i<node->data.layer.width;++i){
			sig_dadz += dadz[i];
			sig_dcda += dcda[i];
		}
		weight_index = 0;
		for (uint64_t i = 0;i<node->prev_count;++i){
			double wa_1 = 0;
			layer* prev = net->nodes[node->prev[i]];
			if (prev->tag == INPUT_NODE){
				for (uint64_t n = 0;n<node->data.layer.width;++n){
					for (uint64_t k = weight_index;k < weight_index + prev->data.layer.width;++k){
						wa_1+= node->data.layer.weights[n][k] * prev->data.input.output[k-weight_index];
					}
				}
			}
			else{
				for (uint64_t n = 0;n<node->data.layer.width;++n){
					for (uint64_t k = weight_index;k < weight_index + prev->data.layer.width;++k){
						wa_1 += node->data.layer.weights[n][k] * prev->data.layer.activated[k-weight_index];
					}
				}
			}
			node->data.layer.prev_weight_gradients[i] += wa_1 * sig_dadz * sig_dcda;
			weight_index += prev->data.layer.width;
		}
		clamp_gradients(net, node->data.layer.prev_weight_gradients, node->prev_count);
	}
	else{
		for (uint64_t i = 0;i<node->data.layer.width;++i){
			uint64_t weight_index = 0;
			for (uint64_t p = 0;p<node->prev_count;++p){
				layer* prev = net->nodes[node->prev[p]];
				if (prev->tag == INPUT_NODE){
					for (uint64_t k = 0;k<prev->data.layer.width;++k){
						double dzdw = prev->data.input.output[k];
						node->data.layer.weight_gradients[i][weight_index] += dzdw * dadz[i] * dcda[i];
						clamp_gradient(net, &node->data.layer.weight_gradients[i][weight_index]);
						weight_index += 1;
					}
					continue;
				}
				for (uint64_t k = 0;k<prev->data.layer.width;++k){
					double dzdw = prev->data.layer.activated[k];
					node->data.layer.weight_gradients[i][weight_index] += dzdw * dadz[i] * dcda[i];
					clamp_gradient(net, &node->data.layer.weight_gradients[i][weight_index]);
					weight_index += 1;
				}
			}
		}
		uint64_t weight_index = 0;
		for (uint64_t i = 0;i<node->prev_count;++i){
			layer* prev = net->nodes[node->prev[i]];
			if (prev->tag == INPUT_NODE){
				weight_index += prev->data.input.width;
				continue;
			}
			for (uint64_t k = 0;k<prev->next_count;++k){
				if (net->nodes[prev->next[k]] != node){
					continue;
				}
				for (uint64_t t = 0;t<prev->data.layer.width;++t){
					for (uint64_t n = 0;n<node->data.layer.width;++n){
						double w = node->data.layer.weights[n][weight_index];
						prev->data.layer.activation_gradients[t] += w * dadz[n] * dcda[n];
					}
					weight_index += 1;
				}
				clamp_gradients(net, prev->data.layer.activation_gradients, prev->data.layer.width);
				break;
			}
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
	assert(node->data.layer.gradient_count != 0);
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		double average = (node->data.layer.bias_gradients[i]/node->data.layer.gradient_count);
		node->data.layer.bias[i] -= net->learning_rate * average;
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
				double average = (node->data.layer.weight_gradients[i][weight_index]/node->data.layer.gradient_count);
				node->data.layer.weights[i][weight_index] -= net->learning_rate * average;
				node->data.layer.weight_gradients[i][weight_index] = 0;
				weight_index += 1;
			}
		}
	}
	if (net->layers_weighted){
		for (uint64_t i = 0;i<node->prev_count;++i){
			double average = (node->data.layer.prev_weight_gradients[i]/node->data.layer.gradient_count);
			node->data.layer.prev_weights[i] -= net->learning_rate * average;
			node->data.layer.prev_weight_gradients[i] = 0;
		}
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		apply_gradients(net, net->nodes[node->next[i]], pass_index);
	}
	node->data.layer.gradient_count = 0;
}

void zero_node_gradients(network* const net, layer* const node){
	node->data.layer.gradient_count = 0;
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
	if (net->layers_weighted){
		for (uint64_t i = 0;i<node->prev_count;++i){
			node->data.layer.prev_weight_gradients[i] = 0;
		}
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
	zero_node_gradients(net, node);
	for (uint64_t i = 0;i<node->next_count;++i){
		zero_gradients(net, net->nodes[node->next[i]], pass_index);
	}
}

void
clear_activation_gradients(network* const net, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			clear_activation_gradients(net, net->nodes[node->next[i]], pass_index);
		}
		return;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.activation_gradients[i] = 0;
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		clear_activation_gradients(net, net->nodes[node->next[i]], pass_index);
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
		double loss = 0;
		for (uint64_t k = i;i<k+net->batch_size;++i){
			memcpy(net->input->data.input.output, data[i], net->input->data.input.width*sizeof(double));
			forward(net, net->input, pass);
			pass += 1;
			loss += losses[net->loss](
				net->loss_output,
				net->output->data.layer.activated,
				expected[i],
				net->output->data.layer.width,
				net->loss_parameter_a
			);
			loss_partials[net->derivative](
				net->output->data.layer.activation_gradients,
				net->output->data.layer.activated,
				expected[i],
				net->output->data.layer.width,
				net->loss_parameter_a
			);
			backward(net, net->output);
			clear_activation_gradients(net, net->input, pass);
			pass += 1;
		}
		apply_gradients(net, net->input, pass);
		pass += 1;
		reset_simulation_flags(net, net->input);
		printf("loss: %lf\n", loss/net->batch_size);
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
	layer_weight_inits[net->layer_weight](
		node->data.layer.prev_weights,
		node->prev_count,
		net->prev_parameter_a,
		net->prev_parameter_b
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
	double a = sqrtf(1/(in_size+out_size));
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++k){
			out[i][k] = uniform_distribution(-a, a);
		}
	}
}

void
weight_initialization_he(double** const out, uint64_t in_size, uint64_t out_size, double aa, double b){
	double a = sqrtf(6/in_size);
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++k){
			out[i][k] = uniform_distribution(-a, a);
		}
	}
}

void
weight_initialization_lecun(double** const out, uint64_t in_size, uint64_t out_size, double a, double b){
	double std = sqrtf(1/in_size);
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
layer_weight_initialization_uniform(double* const buffer, uint64_t size, double a, double b){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = uniform_distribution(a, b);
	}
}

void
layer_weight_initialization_normal(double* const buffer, uint64_t size, double a, double b){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = normal_distribution(a, b);
	}
}

void
layer_weight_initialization_strong(double* const buffer, uint64_t size, double a, double b){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = 1;
	}
}

void
layer_weight_initialization_parametric(double* const buffer, uint64_t size, double a, double b){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = a;
	}
}

void
loss_mse_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		output[i] = 2*(result[i]-expected[i]);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		output[i] = 2*(result[i]-expected[i]);
	}
#endif
}

void
loss_mae_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		double term = result[i]-expected[i];
		output[i] = term/fabsf(term);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		double term = result[i]-expected[i];
		output[i] = term/fabsf(term);
	}
#endif
}

void
loss_mape_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		output[i] = (1/powf(expected[i], 2))*term/fabsf(term);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		output[i] = (1/powf(expected[i], 2))*term/fabsf(term);
	}
#endif
}

void
loss_huber_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		if (term <= a){
			output[i] = term;
			continue;
		}
		output[i] = a*(int)((0<term)-(term<0));
	}
#else
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		if (term <= a){
			output[i] = term;
			continue;
		}
		output[i] = a*(int)((0<term)-(term<0));
	}
#endif
}

void
loss_huber_modified_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	double coef = 1/a;
	for (uint64_t i = 0;i<size;++i){
		double term = result[i]-expected[i];
		if (expected[i]-result[i] <= a){
			output[i] = term*coef;
			continue;
		}
		output[i] = (int)((0<term)-(term-0));
	}
#else
	double coef = 1/a;
	for (uint64_t i = 0;i<size;++i){
		double term = result[i]-expected[i];
		if (expected[i]-result[i] <= a){
			output[i] = term*coef;
			continue;
		}
		output[i] = (int)((0<term)-(term-0));
	}
#endif
}

void
loss_cross_entropy_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		output[i] = (-(expected[i]/result[i]))+((1-expected[i])/(1-result[i]));
	}
#else
	for (uint64_t i = 0;i<size;++i){
		output[i] = (-(expected[i]/result[i]))+((1-expected[i])/(1-result[i]));
	}
#endif
}

void
loss_hinge_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		if (expected[i]*result[i] >= 1){
			output[i] = 0;
		}
		output[i] = -expected[i];
	}
#else
	for (uint64_t i = 0;i<size;++i){
		if (expected[i]*result[i] >= 1){
			output[i] = 0;
		}
		output[i] = -expected[i];
	}
#endif
}

void
activation_sigmoid_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		double fx = 1/(1+expf(-buffer[i]));
		output[i] = fx*(1-fx);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		double fx = 1/(1+expf(-buffer[i]));
		output[i] = fx*(1-fx);
	}
#endif
}

void
activation_relu_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		output[i] = (buffer[i] > 0);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		output[i] = (buffer[i] > 0);
	}
#endif
}

void
activation_tanh_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		output[i] = 1-powf(tanh(buffer[i]), 2);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		output[i] = 1-powf(tanh(buffer[i]), 2);
	}
#endif
}

void
activation_linear_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	memset(output, 1, size*sizeof(double));
#else
	memset(output, 1, size*sizeof(double));
#endif
}

void
activation_relu_leaky_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = 0.01;
	}
#else
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = 0.01;
	}
#endif
}

void
activation_relu_parametric_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a;
	}
#else
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a;
	}
#endif
}

void
activation_elu_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a+(a*(expf(buffer[i])-1));
	}
#else
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a+(a*(expf(buffer[i])-1));
	}
#endif
}

void
activation_softmax_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	double sum = 0.0f;
	double* softmax_values = malloc(sizeof(double)*size);
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
#else
	double sum = 0.0f;
	double* softmax_values = malloc(sizeof(double)*size);
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
#endif
}

void
activation_swish_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	for (uint64_t i = 0;i<size;++i){
		double fx = 1/(1+expf(-buffer[i]*a));
		output[i] = fx+a*buffer[i]*fx*(1-fx);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		double fx = 1/(1+expf(-buffer[i]*a));
		output[i] = fx+a*buffer[i]*fx*(1-fx);
	}
#endif
}

void
activation_gelu_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	double s2op = sqrt(2/M_PI);
	for (uint64_t i = 0;i<size;++i){
		double x = buffer[i];
		double inside = tanh(s2op*(x+GELU_C*powf(x, 3)));
		output[i] = 0.5*(1+inside)+0.5*x*(1-powf(inside, 2))*s2op*(1+3*GELU_C*x*x);
	}
#else
	double s2op = sqrt(2/M_PI);
	for (uint64_t i = 0;i<size;++i){
		double x = buffer[i];
		double inside = tanh(s2op*(x+GELU_C*powf(x, 3)));
		output[i] = 0.5*(1+inside)+0.5*x*(1-powf(inside, 2))*s2op*(1+3*GELU_C*x*x);
	}
#endif
}

void
activation_selu_partial(double* const output, const double* const buffer, uint64_t size, double a){
#ifdef __SSE__
	double lambda_alpha = SELU_LAMBDA*SELU_ALPHA;
	for (uint64_t i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			output[i] = SELU_LAMBDA;
			continue;
		}
		output[i] = lambda_alpha*expf(x);
	}
#else
	double lambda_alpha = SELU_LAMBDA*SELU_ALPHA;
	for (uint64_t i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			output[i] = SELU_LAMBDA;
			continue;
		}
		output[i] = lambda_alpha*expf(x);
	}
#endif
}

double
loss_mse(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	uint64_t i;
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d e = _mm_load_pd(expected+i);
		__m128d r = _mm_load_pd(result+i);
		__m128d loss = _mm_sub_pd(e, r);
		_mm_store_pd(buffer+i,loss);
		__m128d loss_squared = _mm_mul_pd(loss, loss);
		s = _mm_add_pd(s,loss_squared);
	}
	double sum_array[DOUBLE_DIV];
	_mm_store_pd(sum_array, s);
	double sum = sum_array[0]+sum_array[1];
	for (;i<size;++i){
		double loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += loss*loss;
	}
	return (sum)/(size);
#else
	double sum = 0;
	for (uint64_t i = 0;i<size;++i){
		double loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += loss*loss;
	}
	return (sum)/(size);
#endif
}

double
loss_mae(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	uint64_t i;
	for (i= 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d e = _mm_load_pd(expected+i);
		__m128d r = _mm_load_pd(result+i);
		__m128d loss = _mm_sub_pd(e, r);
		_mm_store_pd(buffer+i,loss);
		__m128d abs_loss = _mm_andnot_pd(_mm_set1_pd(-0.f), loss);
		s = _mm_add_pd(s, abs_loss);
	}
	double sum_array[DOUBLE_DIV];
	_mm_store_pd(sum_array, s);
	double sum = sum_array[0]+sum_array[1];
	for (;i<size;++i){
		double loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += fabsf(loss);
	}
	return sum/(size);
#else
	double sum = 0;
	for (uint64_t i = 0;i<size;++i){
		double loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += abs(loss);
	}
	return sum/(size);
#endif
}

double
loss_mape(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	uint64_t i;
	for (i= 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d e = _mm_load_pd(expected+i);
		__m128d r = _mm_load_pd(result+i);
		__m128d loss = _mm_sub_pd(e, r);
		_mm_store_pd(buffer+i,loss);
		__m128d abs_loss = _mm_andnot_pd(_mm_set1_pd(-0.f), loss);
		__m128d loss_div_e = _mm_div_pd(abs_loss, e);
		s = _mm_add_pd(s,loss_div_e);
	}
	double sum_array[4];
	_mm_store_pd(sum_array, s);
	double sum = sum_array[0]+sum_array[1];
	for (;i<size;++i){
		double expect = expected[i];
		double loss = expect-result[i];
		buffer[i] = loss;
		sum += fabsf(loss/expect);
	}
	return sum/(size);

#else
	double sum = 0;
	for (uint64_t i = 0;i<size;++i){
		double expect = expected[i];
		double loss = expect-result[i];
		buffer[i] = loss;
		sum += abs(loss/expect);
	}
	return sum/(size);
#endif
}

double
loss_huber(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	__m128d param_sq_half = _mm_set1_pd(a*a*0.5f);
	__m128d param = _mm_set1_pd(a);
	__m128d half = _mm_set1_pd(0.5f);
	size_t i;
	for (i= 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d e = _mm_load_pd(expected+i);
		__m128d r = _mm_load_pd(result+i);
		__m128d loss = _mm_sub_pd(e, r);
		_mm_store_pd(buffer+i,loss);
		__m128d abs_loss = _mm_andnot_pd(_mm_set1_pd(-0.f), loss);
		__m128d mask = _mm_cmple_pd(abs_loss, param);
		__m128d case1 = _mm_mul_pd(_mm_mul_pd(loss, loss), half);
		__m128d case2 = _mm_sub_pd(_mm_mul_pd(param, abs_loss), param_sq_half);
		__m128d combined = _mm_or_pd(_mm_and_pd(mask, case1), _mm_andnot_pd(mask, case2));
		s = _mm_add_pd(s,combined);
	}
	double sum_array[DOUBLE_DIV];
	_mm_store_pd(sum_array, s);
	double sum = sum_array[0]+sum_array[1];
	double hpsq = a*a*0.5;
	for (;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		double x = expect-res;
		buffer[i] = x;
		if (abs(x) <= a){
			sum += x*x*0.5;
			continue;
		}
		sum += (a*abs(x))-hpsq;
	}
	return sum;

#else
	double sum = 0;
	double hpsq = a*a*0.5;
	for (uint64_t i = 0;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		double x = expect-res;
		buffer[i] = x;
		if (abs(x) <= a){
			sum += x*x*0.5;
			continue;
		}
		sum += (a*abs(x))-hpsq;
	}
	return sum;
#endif
}

double
loss_huber_modified(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	uint64_t i;
	for (i= 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d e = _mm_load_pd(expected+i);
		__m128d r = _mm_load_pd(result+i);
		__m128d loss = _mm_sub_pd(e, r);
		_mm_store_pd(buffer+i,loss);
		__m128d prod = _mm_mul_pd(e, r);
		__m128d mask = _mm_cmpgt_pd(prod, _mm_set1_pd(-1.f));
		__m128d ones = _mm_set1_pd(1.f);
		__m128d sub = _mm_sub_pd(ones, prod);
		__m128d zeros = _mm_setzero_pd();
		__m128d case1 = _mm_max_pd(zeros, sub);
		case1 = _mm_mul_pd(case1, case1);
		__m128d case2 = _mm_mul_pd(_mm_set1_pd(-4.f), prod);
		__m128d combined = _mm_or_pd(_mm_and_pd(mask, case1), _mm_andnot_pd(mask, case2));
		s = _mm_add_pd(s,combined);
	}
	double sum_array[DOUBLE_DIV];
	_mm_store_pd(sum_array, s);
	double sum = sum_array[0]+sum_array[1];
	for (;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		double x = expect-res;
		buffer[i] = x;
		x = expect*res;
		if (x > -1){
			sum += pow(fmax(0, 1-x),2);
			continue;
		}
		sum -= 4*x;
	}
	return sum;

#else
	double sum = 0;
	for (uint64_t i = 0;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		double x = expect-res;
		buffer[i] = x;
		x = expect*res;
		if (x > -1){
			sum += pow(fmax(0, 1-x),2);
			continue;
		}
		sum -= 4*x;
	}
	return sum;
#endif
}

double
loss_cross_entropy(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	uint64_t i;
	for (i= 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d e = _mm_load_pd(expected+i);
		__m128d r = _mm_load_pd(result+i);
		__m128d loss = _mm_sub_pd(e, r);
		_mm_store_pd(buffer+i,loss);
		__m128d log_r = _mm_set_pd(logf(r[1]), logf(r[0]));
		__m128d term = _mm_mul_pd(e, log_r);
		s = _mm_add_pd(s,term);
	}
	double sum_array[DOUBLE_DIV];
	_mm_store_pd(sum_array, s);
	double sum = sum_array[0]+sum_array[1];
	for (;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		buffer[i] = expect-res;
		sum += expect*logf(res);
	}
	return -sum;
#else
	double sum = 0;
	for (uint64_t i = 0;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		buffer[i] = expect-res;
		sum += expect*log(res);
	}
	return -sum;
#endif
}

double
loss_hinge(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	uint64_t i;
	for (i= 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d e = _mm_load_pd(expected+i);
		__m128d r = _mm_load_pd(result+i);
		__m128d loss = _mm_sub_pd(e, r);
		_mm_store_pd(buffer+i,loss);
		__m128d prod = _mm_mul_pd(e, r);
		__m128d ones = _mm_set1_pd(1.0f);
		__m128d term = _mm_sub_pd(ones, prod);
		__m128d zeros = _mm_setzero_pd();
		__m128d hinge = _mm_max_pd(zeros, term);
		s = _mm_add_pd(s,hinge);
	}
	double sum_array[DOUBLE_DIV];
	_mm_store_pd(sum_array, s);
	double sum = sum_array[0]+sum_array[1];
	for (;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		buffer[i] = expect-res;
		sum += fmaxf(0,1-(expect*res));
	}
	return sum;
#else
	double sum = 0;
	for (uint64_t i = 0;i<size;++i){
		double expect = expected[i];
		double res = result[i];
		buffer[i] = expect-res;
		sum += fmax(0,1-(expect*res));
	}
	return sum;
#endif
}

#ifdef __SSE__
__m128d
exp_neg_pd(__m128d x) {
	const __m128d one = _mm_set1_pd(1.0f);
	const __m128d exp_c1 = _mm_set1_pd(0.04166669f);
	const __m128d exp_c2 = _mm_set1_pd(0.5000004f);
	__m128d x2 = _mm_mul_pd(x, x);  // x^2
	__m128d x3 = _mm_mul_pd(x2, x); // x^3
	// Compute the polynomial approximation: e^-x ≈ 1 - x - x^2/2 - x^3/6
	__m128d poly = _mm_sub_pd(one, x);        // 1 - x
	poly = _mm_sub_pd(poly, _mm_mul_pd(exp_c2, x2)); // 1 - x - x^2/2
	poly = _mm_sub_pd(poly, _mm_mul_pd(exp_c1, x3)); // 1 - x - x^2/2 - x^3/6
	return poly;
}
#endif

void
activation_sigmoid(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	uint64_t i;
	const __m128d one = _mm_set1_pd(1.0f);
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		x = _mm_xor_pd(x, _mm_set1_pd(-0.f));
		__m128d exp_neg_x = exp_neg_pd(x);
		__m128d sigmoid = _mm_div_pd(one, _mm_add_pd(one, exp_neg_x));
		_mm_store_pd(buffer+i, sigmoid);
	}
	for (;i<size;++i){
		buffer[i] = 1/(1+expf(-output[i]));
	}
#else
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = 1/(1+expf(-output[i]));
	}
#endif
}

void
activation_relu(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	uint64_t i;
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d zeros = _mm_setzero_pd();
		__m128d term = _mm_load_pd(output+i);
		__m128d relu = _mm_max_pd(zeros, term);
		_mm_store_pd(buffer+i, relu);
	}
	for (;i<size;++i){
		buffer[i] = fmaxf(0,output[i]);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = fmax(0,output[i]);
	}
#endif
}

#ifdef __SSE__
__m128d
tanh_pd(__m128d x) {
	const __m128d one = _mm_set1_pd(1.0f);
	// calculate e^(2x)
	__m128d exp2x = exp_neg_pd(_mm_mul_pd(x, _mm_set1_pd(-2.0f)));
	exp2x = _mm_div_pd(one, exp2x);
	// calculate (e^(2x) - 1) / (e^(2x) + 1)
	__m128d num = _mm_sub_pd(exp2x, one);
	__m128d den = _mm_add_pd(exp2x, one);
	__m128d tanh_x = _mm_div_pd(num, den);
	return tanh_x;
}
#endif

void
activation_tanh(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	uint64_t i;
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d tanh_approx = tanh_pd(x);
		_mm_store_pd(buffer+i, tanh_approx);
	}
	for (;i<size;++i){
		buffer[i] = tanh(output[i]);
	}
#else
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = tanh(output[i]);
	}
#endif
}

void
activation_binary_step(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	const __m128d zero = _mm_setzero_pd();
	uint64_t i;
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d step = _mm_cmpge_pd(x, zero);
		_mm_store_pd(buffer+i, step);
	}
	for (;i<size;++i){
		buffer[i] = output[i] >= 0;
	}
#else
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = output[i] >= 0;
	}
#endif
}

void
activation_linear(double* const buffer, const double* const output, uint64_t size, double a){
	memcpy(buffer, output, size*sizeof(double));
}

void
activation_relu_leaky(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	uint64_t i;
	const __m128d tenth = _mm_set1_pd(0.1f);
	for (i=0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d term = _mm_max_pd(_mm_mul_pd(tenth, x), x);
		_mm_store_pd(buffer+i, term);
	}
	for (;i<size;++i){
		double x = output[i];
		buffer[i] = fmaxf(0.1*x,x);
	}
#else
	double x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = fmax(0.1*x,x);
	}
#endif
}

void
activation_relu_parametric(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	uint64_t i;
	const __m128d tenth = _mm_set1_pd(a);
	for (i=0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d term = _mm_max_pd(_mm_mul_pd(tenth, x), x);
		_mm_store_pd(buffer+i, term);
	}
	for (;i<size;++i){
		double x = output[i];
		buffer[i] = fmaxf(a*x,x);
	}
#else
	double x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = fmax(a*x, x);
	}
#endif
}

void
activation_elu(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	const __m128d zero = _mm_setzero_pd();
	const __m128d one = _mm_set1_pd(1.0f);
	const __m128d alpha = _mm_set1_pd(a);
	uint64_t i;
	for (i=0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d mask = _mm_cmplt_pd(x, zero);
		__m128d negs = _mm_and_pd(mask, _mm_mul_pd(alpha, _mm_sub_pd(exp_neg_pd(x), one)));
		__m128d term = _mm_add_pd(_mm_andnot_pd(mask, x), negs);
		_mm_store_pd(buffer+i, term);
	}
	for (;i<size;++i){
		double x = output[i];
		if (x < 0){
			buffer[i] = a*(expf(x)-1);
		}
		else{
			buffer[i] = x;
		}
	}
#else
	double x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		if (x < 0){
			buffer[i] = a*(expf(x)-1);
		}
		else{
			buffer[i] = x;
		}
	}
#endif
}

void
activation_softmax(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	__m128d s = _mm_setzero_pd();
	const __m128d n1 = _mm_set1_pd(-1.0f);
	uint64_t i;
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d exp_x = exp_neg_pd(_mm_mul_pd(x, n1));
		s = _mm_add_pd(s, exp_x);
	}
	double simd_s[DOUBLE_DIV];
	_mm_store_pd(simd_s, s);
	double denom = simd_s[0]+simd_s[1];
	for (;i<size;++i){
		denom += expf(output[i]);
	}
	const __m128d d = _mm_set1_pd(denom);
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d exp_x = exp_neg_pd(_mm_mul_pd(x, n1));
		__m128d term = _mm_div_pd(exp_x, d);
		_mm_store_pd(buffer+i, term);
	}
	for (;i<size;++i){
		buffer[i] = expf(output[i])/denom;
	}
#else
	double denom = 0;
	for (uint64_t i = 0;i<size;++i){
		denom += expf(output[i]);
	}
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = expf(output[i])/denom;
	}
#endif
}

void
activation_swish(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	uint64_t i;
	const __m128d one = _mm_set1_pd(1.0f);
	for (i = 0;i+DOUBLE_DIV < size;i += DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d denom = _mm_add_pd(one, exp_neg_pd(x));
		__m128d term = _mm_div_pd(x, denom);
		_mm_store_pd(buffer+i, term);
	}
	for (;i<size;++i){
		double x = output[i];
		buffer[i] = x/(1+expf(-x));
	}
#else
	double x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = x/(1+expf(-x));
	}
#endif
}

void
activation_gelu(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	uint64_t i;
	const double s2p = sqrtf(2.0f)/M_PI;
	const __m128d half = _mm_set1_pd(0.5f);
	const __m128d one = _mm_set1_pd(1.0f);
	const __m128d sqrt2vpi = _mm_set1_pd(s2p);
	const __m128d gelu_c = _mm_set1_pd(GELU_C);
	for (i = 0 ;i+DOUBLE_DIV <= size;i += DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d a = _mm_mul_pd(
			sqrt2vpi,
			_mm_add_pd(
				x,
				_mm_mul_pd(
					gelu_c,
					_mm_mul_pd(x, _mm_mul_pd(x, x))
				)
			)
		);
		__m128d tanh_a = tanh_pd(a);
		__m128d gelu = _mm_mul_pd(
			half,
			_mm_mul_pd(x, _mm_add_pd(one, tanh_a))
		);
		_mm_store_pd(buffer+i, gelu);
	}
	for (;i<size;++i){
		double x = output[i];
		buffer[i] = 0.5*x*(1+tanh(s2p*(x+(GELU_C*pow(x,3)))));
	}
#else
	double x;
	const double s2p  = sqrtf(2/M_PI);
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = 0.5*x*(1+tanh(s2p*(x+(GELU_C*pow(x,3)))));
	}
#endif
}

void
activation_selu(double* const buffer, const double* const output, uint64_t size, double a){
#ifdef __SSE__
	const __m128d lam = _mm_set1_pd(SELU_LAMBDA);
	const __m128d alf = _mm_set1_pd(SELU_ALPHA);
	const __m128d one = _mm_set1_pd(1.0f);
	const __m128d lamalf = _mm_mul_pd(lam, alf);
	const __m128d zero = _mm_setzero_pd();
	uint64_t i;
	for (i = 0;i+DOUBLE_DIV<=size;i+=DOUBLE_DIV){
		__m128d x = _mm_load_pd(output+i);
		__m128d xl = _mm_mul_pd(x, lam);
		_mm_store_pd(buffer+i, xl);
		__m128d mask = _mm_cmpgt_pd(zero, x);
		__m128d negs = _mm_and_pd(mask, _mm_mul_pd(lamalf, _mm_sub_pd(exp_neg_pd(x), one)));
		__m128d term = _mm_add_pd(_mm_andnot_pd(mask, x), negs);
		_mm_store_pd(buffer+i, term);
	}
	double lambda_alpha = SELU_LAMBDA*SELU_ALPHA;
	for (i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			buffer[i] = SELU_LAMBDA * x;
			continue;
		}
		buffer[i] = lambda_alpha*(expf(x)-1);
	}
#else
	double lambda_alpha = SELU_LAMBDA*SELU_ALPHA;
	for (uint64_t i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			buffer[i] = SELU_LAMBDA * x;
			continue;
		}
		buffer[i] = lambda_alpha*(expf(x)-1);
	}
#endif
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
		fwrite(&node->prev_count, sizeof(uint64_t), 1, outfile);
		fwrite(&node->data.layer.prev_weights, sizeof(double), node->prev_count, outfile);
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
#ifdef __SSE__
		node->data.layer.output = pool_request_aligned(mem, sizeof(double)*node->data.layer.width, SSE_ALIGNMENT);
#else
		node->data.layer.output = pool_request(mem, sizeof(double)*node->data.layer.width);
#endif
		if (node->tag == LAYER_NODE){
			uint64_t sum;
			fread(&node->prev_count, sizeof(uint64_t), 1, infile);
#ifdef __SSE__
			node->data.layer.prev_weights = pool_request_aligned(mem, sizeof(double)*node->prev_count, SSE_ALIGNMENT);
#else
			node->data.layer.prev_weights = pool_request(mem, sizeof(double)*node->prev_count);
#endif
			fread(&node->data.layer.prev_weights, sizeof(double), node->prev_count, infile);
			fread(&sum, sizeof(uint64_t), 1, infile);
#ifdef __SSE__
			node->data.layer.weights = pool_request_aligned(mem, sizeof(double*)*node->data.layer.width, SSE_ALIGNMENT);
			node->data.layer.weight_gradients = pool_request_aligned(mem, sizeof(double*)*node->data.layer.width, SSE_ALIGNMENT);
#else
			node->data.layer.weights = pool_request(mem, sizeof(double*)*node->data.layer.width);
			node->data.layer.weight_gradients = pool_request(mem, sizeof(double*)*node->data.layer.width);
#endif
			for (uint64_t i = 0;i<node->data.layer.width;++i){
#ifdef __SSE__
				node->data.layer.weights[i] = pool_request_aligned(mem, sizeof(double)*sum, SSE_ALIGNMENT);
				node->data.layer.weight_gradients[i] = pool_request_aligned(mem, sizeof(double)*sum, SSE_ALIGNMENT);
#else
				node->data.layer.weights[i] = pool_request(mem, sizeof(double)*sum);
				node->data.layer.weight_gradients[i] = pool_request(mem, sizeof(double)*sum);
#endif
				fread(&node->data.layer.weights[i], sizeof(double), sum, infile);
			}
#ifdef __SSE__
			node->data.layer.bias = pool_request_aligned(mem, sizeof(double)*node->data.layer.width, SSE_ALIGNMENT);
#else
			node->data.layer.bias = pool_request(mem, sizeof(double)*node->data.layer.width);
#endif
			fread(&node->data.layer.bias, sizeof(double), node->data.layer.width, infile);
			fread(&node->data.layer.activation, sizeof(ACTIVATION_FUNC), 1, infile);
			fread(&node->data.layer.derivative, sizeof(ACTIVATION_PARTIAL_FUNC), 1, infile);
			fread(&node->data.layer.parameter_a, sizeof(uint64_t), 1, infile);
#ifdef __SSE__
			node->data.layer.activated = pool_request_aligned(mem, sizeof(double)*node->data.layer.width, SSE_ALIGNMENT);
			node->data.layer.bias_gradients = pool_request_aligned(mem, sizeof(double)*node->data.layer.width, SSE_ALIGNMENT);
			node->data.layer.activation_gradients = pool_request_aligned(mem, sizeof(double)*node->data.layer.width, SSE_ALIGNMENT);
#else
			node->data.layer.activated = pool_request(mem, sizeof(double)*node->data.layer.width);
			node->data.layer.bias_gradients = pool_request(mem, sizeof(double)*node->data.layer.width);
			node->data.layer.activation_gradients = pool_request(mem, sizeof(double)*node->data.layer.width);
#endif
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
		node->data.layer.gradient_count = 0;
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
	fwrite(&net->layer_weight, sizeof(LAYER_WEIGHT_FUNC), 1, outfile);
	fwrite(&net->batch_size, sizeof(uint64_t), 1, outfile);
	fwrite(&net->learning_rate, sizeof(double), 1, outfile);
	fwrite(&net->loss_parameter_a, sizeof(double), 1, outfile);
	fwrite(&net->weight_parameter_a, sizeof(double), 1, outfile);
	fwrite(&net->weight_parameter_b, sizeof(double), 1, outfile);
	fwrite(&net->bias_parameter_a, sizeof(double), 1, outfile);
	fwrite(&net->bias_parameter_b, sizeof(double), 1, outfile);
	fwrite(&net->node_capacity, sizeof(uint64_t), 1, outfile);
	fwrite(&net->prev_parameter_a, sizeof(double), 1, outfile);
	fwrite(&net->prev_parameter_b, sizeof(double), 1, outfile);
	fwrite(&net->gradient_clamp, sizeof(double), 1, outfile);
	fwrite(&net->prune, sizeof(ACTIVATION_FUNC), 1, outfile);
	fwrite(&net->prune_parameter_a, sizeof(double), 1, outfile);
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
	fread(&net.layer_weight, sizeof(LAYER_WEIGHT_FUNC), 1, infile);
	fread(&net.batch_size, sizeof(uint64_t), 1, infile);
	fread(&net.learning_rate, sizeof(double), 1, infile);
	fread(&net.loss_parameter_a, sizeof(double), 1, infile);
	fread(&net.weight_parameter_a, sizeof(double), 1, infile);
	fread(&net.weight_parameter_b, sizeof(double), 1, infile);
	fread(&net.bias_parameter_a, sizeof(double), 1, infile);
	fread(&net.bias_parameter_b, sizeof(double), 1, infile);
	fread(&net.node_capacity, sizeof(uint64_t), 1, infile);
	fread(&net.prev_parameter_a, sizeof(double), 1, infile);
	fread(&net.prev_parameter_b, sizeof(double), 1, infile);
	fread(&net.gradient_clamp, sizeof(double), 1, infile);
	fread(&net.prune, sizeof(ACTIVATION_FUNC), 1, infile);
	fread(&net.prune_parameter_a, sizeof(double), 1, infile);
	net.nodes = pool_request(mem, sizeof(uint64_t)*net.node_capacity);
	load_nodes(&net, mem, infile);
	fclose(infile);
#ifdef __SSE__
	net.loss_output = pool_request_aligned(mem, sizeof(double)*net.input->data.input.width, SSE_ALIGNMENT);
#else
	net.loss_output = pool_request(mem, sizeof(double)*net.input->data.input.width);
#endif
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
		else {
			if (node->next_count == 0){
				printf("(output)\n");
			}
			printf("gradient_count: %lu\n", node->data.layer.gradient_count);
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

void
network_prune(network* const net){
	for (uint64_t i = 0;i<net->node_count;++i){
		layer* node = net->nodes[i];
		if (node == net->input){
			continue;
		}
		activations[net->prune](
			node->data.layer.prev_weights,
			node->data.layer.prev_weights,
			node->prev_count,
			net->prune_parameter_a
		);
	}
}

void
network_compose_layer(network* const net, layer* const new){
	uint64_t id = network_register_layer(net, new);
	allocate_node_weights(net, net->mem, new);
	zero_node_gradients(net, new);
	new->pass_index = net->input->pass_index;
	for (uint64_t i = 0;i<net->node_count-1;++i){
		layer* node = net->nodes[i];
		if (node != net->output){
			layer_link(net, net->mem, i, id);
			update_layer_connection_data(net, new, i);
		}
		if (node != net->input){
			layer_link(net, net->mem, id, i);
			update_layer_connection_data(net, node, id);
		}
	}
}

void
update_layer_connection_data(network* const net, layer* const node, uint64_t target_id){
	uint64_t target_weight_index = 0;
	for (uint64_t i = 0;i<node->prev_count;++i){
		if (node->prev[i] == target_id){
			break;
		}
		target_weight_index += net->nodes[node->prev[i]]->data.layer.width;
	}
	uint64_t newsize = target_weight_index + net->nodes[target_id]->data.layer.width;
	for (uint64_t i = 0;i<node->data.layer.width;++i){
#ifdef __SSE__
		double* new_weights = pool_request_aligned(net->mem, sizeof(double)*newsize, SSE_ALIGNMENT);
		double* new_weight_gradients = pool_request_aligned(net->mem, sizeof(double)*newsize, SSE_ALIGNMENT);
#else
		double* new_weights = pool_request(net->mem, sizeof(double)*newsize);
		double* new_weight_gradients = pool_request(net->mem, sizeof(double)*newsize);
#endif
		memset(new_weight_gradients, 0, newsize*sizeof(double));
		weight_inits[net->weight](
			&new_weights,
			newsize, 1,
			net->weight_parameter_a,
			net->weight_parameter_b
		);
		memcpy(new_weights, node->data.layer.weights[i], target_weight_index*sizeof(double));
		node->data.layer.weight_gradients[i] = new_weight_gradients;
		node->data.layer.weights[i] = new_weights;
	}
#ifdef __SSE__
	double* new_prev_weights = pool_request_aligned(net->mem, sizeof(double)*node->prev_count, SSE_ALIGNMENT);
	double* new_prev_weight_gradients = pool_request_aligned(net->mem, sizeof(double)*node->prev_count, SSE_ALIGNMENT);
#else
	double* new_prev_weights = pool_request(net->mem, sizeof(double)*node->prev_count);
	double* new_prev_weight_gradients = pool_request(net->mem, sizeof(double)*node->prev_count);
#endif
	memset(new_prev_weight_gradients, 0, node->prev_count*sizeof(double));
	memcpy(new_prev_weights, node->data.layer.prev_weights, (node->prev_count-1)*sizeof(double));
	layer_weight_inits[net->layer_weight](
		new_prev_weights,
		1,
		net->prev_parameter_a,
		net->prev_parameter_b
	);
	node->data.layer.prev_weights = new_prev_weights;
	node->data.layer.prev_weight_gradients = new_prev_weight_gradients;
}

void
grow_network(network* const net, double** training_data, uint64_t samples, double** expected, uint64_t epochs, uint64_t prune_epoch, uint64_t grow_epoch){
	uint64_t in = network_register_layer(net, net->input);
	uint64_t out = network_register_layer(net, net->output);
	layer_link(net, net->mem, in, out);
	network_build(net);
	net->layers_weighted = 1;
	layer* initial = grow_layer(net->mem);
	network_compose_layer(net, initial);
	for (uint64_t i = 0;i<epochs;++i){
		network_train(net, training_data, samples, expected);
		if ((i+1)%prune_epoch == 0){
			network_prune(net);
			if ((i+1)%grow_epoch == 0){
				layer* new = grow_layer(net->mem);
				network_compose_layer(net, new);
			}
		}
	}
	network_prune(net);
}

void
grow_network_retrain(network* const net, double** training_data, uint64_t samples, double** expected, uint64_t epochs, uint64_t prune_epoch, uint64_t grow_epoch){
	uint64_t in = network_register_layer(net, net->input);
	uint64_t out = network_register_layer(net, net->output);
	layer_link(net, net->mem, in, out);
	network_build(net);
	net->layers_weighted = 1;
	layer* initial = grow_layer(net->mem);
	uint64_t initial_id = network_register_layer(net, initial);
	for (uint64_t i = 0;i<net->node_count-1;++i){
		if (net->nodes[i] != net->output){
			layer_link(net, net->mem, i, initial_id);
		}
		if (net->nodes[i] != net->input){
			layer_link(net, net->mem, initial_id, i);
		}
	}
	network_build(net);
	for (uint64_t e = 0;e<epochs;++e){
		network_train(net, training_data, samples, expected);
		if ((e+1)%prune_epoch == 0){
			network_prune(net);
		}
		if ((e+1) % grow_epoch == 0){
			layer* new = grow_layer(net->mem);
			uint64_t new_id = network_register_layer(net, new);
			for (uint64_t i = 0;i<net->node_count;++i){
				if (net->nodes[i] != net->output){
					layer_link(net, net->mem, i, new_id);
				}
				if (net->nodes[i] != net->input){
					layer_link(net, net->mem, new_id, i);
				}
			}
			network_build(net);
		}
	}
}

layer*
grow_layer(pool* const mem){
	ACTIVATION_FUNC f = random()%ACTIVATION_COUNT;
	double a = 0;
	switch (f){
		case ACTIVATION_RELU_PARAMETRIC:
		case ACTIVATION_ELU:
			a = (random()%10)/5;
		default:
			break;
	}
	uint64_t width = (random() % 16) * 16;
	return layer_init(mem, width, f, a);
}

int
main(int argc, char** argv){
	set_seed(time(NULL));
	pool mem = pool_alloc(TEMP_POOL_SIZE, POOL_STATIC);
	layer* input = input_init(&mem, 8);
	layer* output = layer_init(&mem, 8, ACTIVATION_SIGMOID, 0);
	network net = network_init(
		&mem,
		input, output,
		WEIGHT_INITIALIZATION_XAVIER,
		BIAS_INITIALIZATION_ZERO,
		LAYER_WEIGHT_INITIALIZATION_STRONG,
		ACTIVATION_RELU,
		1, 2,
		0, 0,
		0, 0,
		0,
		16, 0.001,
		5,
		LOSS_MSE
	);
	uint64_t samples = 512;
	double** training_data = pool_request(&mem, sizeof(double*)*samples);
	uint8_t pos = 0;
	for (uint32_t i = 0;i<samples;++i){
		training_data[i] = pool_request(&mem, sizeof(double)*net.input->data.input.width);
		for (uint32_t k = 0;k<net.input->data.input.width;++k){
			training_data[i][k] = 0;
		}
		training_data[i][pos] = 1;
		pos += 1;
		if (pos == 8){
			pos = 0;
		}
	}
	pos = 0;
	double** expected = pool_request(&mem, sizeof(double*)*samples);
	for (uint32_t i = 0;i<samples;++i){
		expected[i] = pool_request(&mem, sizeof(double)*net.output->data.layer.width);
		for (uint32_t k = 0;k<net.output->data.layer.width;++k){
			expected[i][k] = 0;
		}
		expected[i][pos] = 1;
		pos += 1;
		if (pos == 8){
			pos = 0;
		}
	}
	grow_network_retrain(
		&net,
		training_data, samples, expected,
		1000, 100, 500
	);
	prediction_vector vec = predict_vector_batched(&net, &mem, &training_data, 1, net.batch_size, net.input->data.input.width);
	printf("predicted %lu (%lf) \n", vec.class[0], vec.probability[0]);
	pool_dealloc(&mem);
	return 0;
}
