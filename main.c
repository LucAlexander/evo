#include "evo.h"
#include "kickstart.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

network
network_init(pool* const mem, uint64_t input, uint64_t output, weight_init weight, bias_init bias, uint64_t batch_size, double learning_rate, loss_function l, loss_derivative ld){
	network net = {
		.mem = mem,
		.temp = pool_alloc(TEMP_POOL_SIZE, POOL_STATIC),
		.input = input_init(mem, input),
		.output = layer_init(mem, output),
		.loss_output = pool_request(mem, sizeof(double)*output),
		.loss = l,
		.derivative=ld,
		.bias = bias,
		.weight = weight,
		.batch_size = batch_size,
		.learning_rate = learning_rate
	};
	layer_link(mem, net.input, net.output);
	return net;
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
	input->next = pool_request(mem, 2*sizeof(layer*));
	input->next_count = 0;
	input->next_capacity = 2;
	input->simulated = 0;
	input->pass_index = 0;
	return input;
}

layer*
layer_init(pool* const mem, uint64_t width){
	layer* node = pool_request(mem, sizeof(layer));
	node->tag = LAYER_NODE;
	node->data.layer.output = pool_request(mem, sizeof(double)*width);
	node->data.layer.activated = pool_request(mem, sizeof(double)*width);
	node->data.layer.width = width;
	node->data.layer.weights = NULL;
	node->data.layer.bias = pool_request(mem, sizeof(double)*width);
	node->data.layer.bias_gradients = pool_request(mem, sizeof(double)*width);
	node->data.layer.activation_gradients = pool_request(mem, sizeof(double)*width);
	node->data.layer.activation = NULL;
	node->prev = pool_request(mem, 2*sizeof(layer*));
	node->prev_count = 0;
	node->prev_capacity = 2;
	node->next = pool_request(mem, 2*sizeof(layer*));
	node->next_count = 0;
	node->next_capacity = 2;
	node->simulated = 0;
	node->pass_index = 0;
	return node;
}

void
layer_link(pool* const mem, layer* const a, layer* const b){
	assert(b->tag != INPUT_NODE);
	if (a->next_count == a->next_capacity){
		layer** new = pool_request(mem, sizeof(layer*)*a->next_capacity*2);
		for (uint64_t i = 0;i<a->next_capacity;++i){
			new[i] = a->next[i];
		}
		a->next_capacity *= 2;
	}
	a->next[a->next_count] = b;
	a->next_count += 1;
	if (b->prev_count == b->prev_capacity){
		layer** new = pool_request(mem, sizeof(layer*)*b->prev_capacity*2);
		for (uint64_t i = 0;i<b->prev_capacity;++i){
			new[i] = b->next[i];
		}
		b->prev_capacity *= 2;
	}
	b->prev[b->prev_count] = b;
	b->prev_count += 1;
}

void
layer_unlink(layer* const a, layer* const b){
	assert(b->tag != INPUT_NODE);
	uint8_t found = 0;
	for (uint64_t i = 0;i<a->next_count;++i){
		if (a->next[i] == b){
			found = 1;
		}
		else if (found == 1){
			a->next[i-1] = a->next[i];
		}
	}
	assert(found == 1);
	a->next_count -= 1;
	found = 0;
	for (uint64_t i = 0;i<b->prev_count;++i){
		if (b->prev[i] == a){
			found = 1;
		}
		else if (found == 1){
			b->prev[i-1] = b->prev[i];
		}
	}
	assert(found == 1);
	b->prev_count -= 1;
}

void
layer_insert(pool* const mem, layer* const a, layer* const b, layer* const c){
	layer_unlink(a, b);
	layer_link(mem, a, b);
	layer_link(mem, b, c);
}

void
reset_simulation_flags(layer* const node){
	if (node->simulated == 0){
		return;
	}
	node->simulated = 0;
	for (uint64_t i = 0;i<node->next_count;++i){
		reset_simulation_flags(node->next[i]);
	}
}

void
allocate_weights(pool* const mem, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			allocate_weights(mem, node->next[i], pass_index);
		}
		return;
	}
	node->data.layer.weights = pool_request(mem, sizeof(double*)*node->data.layer.width);
	node->data.layer.weight_gradients = pool_request(mem, sizeof(double)*node->data.layer.width);
	uint64_t sum = 0;
	for (uint64_t i = 0;i<node->prev_count;++i){
		layer* prev = node->prev[i];
		sum += prev->data.layer.width;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.weights[i] = pool_request(mem, sizeof(double)*sum);
		node->data.layer.weight_gradients[i] = pool_request(mem, sizeof(double)*sum);
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		allocate_weights(mem, node->next[i], pass_index);
	}
}

void
forward(layer* const node, uint64_t pass_index){
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			forward(node->next[i], pass_index);
		}
		return;
	}
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.output[i] = node->data.layer.bias[i];
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = node->prev[p];
			if (prev->simulated == 0){
				weight_index += prev->data.layer.width;
				continue;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				node->data.layer.output[i] += prev->data.layer.activated[k] * node->data.layer.weights[i][weight_index];
				weight_index += 1;
			}
		}
	}
	node->data.layer.activation(node->data.layer.activated, node->data.layer.output);
	node->simulated = 1;
	for (uint64_t i = 0;i<node->next_count;++i){
		forward(node->next[i], pass_index);
	}
}

void backward(network* const net, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		return;
	}
	pool_empty(&net->temp);
	double* dadz = pool_request(&net->temp, sizeof(double)*node->data.layer.width);
	double* dcda = node->data.layer.activation_gradients;
	node->data.layer.derivative(dadz, node->data.layer.output);
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.bias_gradients[i] += 1 * dadz[i] * dcda[i];
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = node->prev[p];
			if (prev->simulated == 0){
				weight_index += prev->data.layer.width;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				double dzdw = prev->data.layer.activated[k];
				node->data.layer.weight_gradients[i][weight_index] += dzdw * dadz[i] * dcda[i];
				weight_index += 1;
			}
		}
	}
	for (uint64_t i = 0;i<node->prev_count;++i){
		layer* prev = node->prev[i];
		if (prev->tag == INPUT_NODE){
			continue;
		}
		for (uint64_t k = 0;k<prev->next_count;++k){
			if (prev->next[k] != node){
				continue;
			}
			for (uint64_t t = 0;t<prev->data.layer.width;++t){
				for (uint64_t n = 0;n<node->data.layer.width;++n){
					prev->data.layer.activation_gradients[t] += node->data.layer.weights[n][i] * dadz[n] * dcda[n];
				}
			}
			break;
		}
	}
	for (uint64_t i = 0;i<node->prev_count;++i){
		backward(net, node->prev[i], pass_index);
	}
}

void
apply_gradients(network* const net, layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		return;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.bias[i] += net->learning_rate * (node->data.layer.bias_gradients[i]/net->batch_size);
		node->data.layer.bias_gradients[i] = 0;
		node->data.layer.activation_gradients[i] = 0;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = node->prev[p];
			if (prev->simulated == 0){
				weight_index += prev->data.layer.width;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				node->data.layer.weights[i][weight_index] += net->learning_rate * (node->data.layer.weight_gradients[i][weight_index]/net->batch_size);
				node->data.layer.weight_gradients[i][weight_index] = 0;
				weight_index += 1;
			}
		}
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		apply_gradients(net, node->next[i], pass_index);
	}
}

void
zero_gradients(layer* const node, uint64_t pass_index){
	if (node->pass_index >= pass_index){
		return;
	}
	node->pass_index += 1;
	if (node->tag == INPUT_NODE){
		return;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		node->data.layer.bias_gradients[i] = 0;
		node->data.layer.activation_gradients[i] = 0;
	}
	for (uint64_t i = 0;i<node->data.layer.width;++i){
		uint64_t weight_index = 0;
		for (uint64_t p = 0;p<node->prev_count;++p){
			layer* prev = node->prev[p];
			if (prev->simulated == 0){
				weight_index += prev->data.layer.width;
			}
			for (uint64_t k = 0;k<prev->data.layer.width;++k){
				node->data.layer.weight_gradients[i][weight_index] = 0;
				weight_index += 1;
			}
		}
	}
	for (uint64_t i = 0;i<node->next_count;++i){
		zero_gradients(node->next[i], pass_index);
	}
}

void network_train(network* const net, double** data, uint64_t data_size, double** expected){
	assert(data_size % net->batch_size == 0);
	uint64_t pass = net->input->pass_index+1;
	zero_gradients(net->input, pass);
	pass += 1;
	for (uint64_t i = 0;i<data_size;++i){
		for (uint64_t k = i;i<k+net->batch_size;++i){
			memcpy(net->input->data.input.output, data[i], net->input->data.input.width);
			forward(net->input, pass);
			pass += 1;
			net->loss(net->loss_output, net->output->data.layer.activated);
			net->derivative(net->output->data.layer.activation_gradients, net->output->data.layer.activated, expected[i]);
			backward(net, net->output, pass);
			pass += 1;
		}
		apply_gradients(net, net->output, pass);
		pass += 1;
	}
}

int main(int argc, char** argv){
	return 0;
}
