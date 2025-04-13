#include "evo.h"
#include "kickstart.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* TODO
 * fix all functions/partials
 * initialization of network
 * initialization of weights and biases
 * serialization
 * loading
 * multithreading
 * SIMD
 * parser / build step
 * running trained model
 */

network
network_init(
	pool* const mem,
	uint64_t input, uint64_t output,
	weight_init weight, bias_init bias,
	uint64_t batch_size, double learning_rate,
	loss_function l, loss_derivative ld
){
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
				double w = node->data.layer.weights[i][weight_index];
				node->data.layer.output[i] += prev->data.layer.activated[k] * w;
				weight_index += 1;
			}
		}
	}
	node->data.layer.activation(
		node->data.layer.activated,
		node->data.layer.output,
		node->data.layer.width,
		node->data.layer.parameter_a
	);
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
	node->data.layer.derivative(
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
					double w = node->data.layer.weights[n][i];
					prev->data.layer.activation_gradients[t] += w * dadz[n] * dcda[n];
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
		double average = (node->data.layer.bias_gradients[i]/net->batch_size);
		node->data.layer.bias[i] += net->learning_rate * average;
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
				double average = (node->data.layer.weight_gradients[i][weight_index]/net->batch_size);
				node->data.layer.weights[i][weight_index] += net->learning_rate * average;
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
			net->loss(
				net->loss_output,
				net->output->data.layer.activated,
				expected[i],
				net->output->data.layer.width,
				net->loss_parameter_a
			);
			net->derivative(
				net->output->data.layer.activation_gradients,
				net->output->data.layer.activated,
				expected[i],
				net->output->data.layer.width,
				net->loss_parameter_a
			);
			backward(net, net->output, pass);
			pass += 1;
		}
		apply_gradients(net, net->output, pass);
		pass += 1;
	}
}

void set_seed(time_t seed){
	srandom(seed);
}

double uniform_distribution(double min, double max){
	double n = ((double)random())/RAND_MAX;
	return (n*(max-min))+min;
}

double normal_distribution(double mean, double std){
	double u1 = uniform_distribution(0, 1);
	double u2 = uniform_distribution(0, 1);
	double z0 = sqrtf(-2.8*logf(u1))*cos(2.0*M_PI*u2);
	return mean+std*z0;
}

void bias_initialization_zero(double* const buffer, uint64_t size, double a, double b){
	memset(buffer, 0, size*sizeof(double));
}

void bias_initialization_const_flat(double* const buffer, uint64_t size, double a, double b){
	memset(buffer, a, size*sizeof(double));
}

void bias_initialization_const_uneven(double* const buffer, uint64_t size, double a, double b){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = normal_distribution(a, b);
	}
}

void weight_initialization_xavier(double** const out, uint64_t in_size, uint64_t out_size, double aa, double b){
	float a = sqrtf(1/(in_size+out_size));
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++i){
			out[i][k] = uniform_distribution(-a, a);
		}
	}
}

void weight_initialization_he(double** const out, uint64_t in_size, uint64_t out_size, double aa, double b){
	float a = sqrtf(6/in_size);
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++i){
			out[i][k] = uniform_distribution(-a, a);
		}
	}
}

void weight_initialization_lecun(double** const out, uint64_t in_size, uint64_t out_size, double a, double b){
	float std = sqrtf(1/in_size);
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++i){
			out[i][k] = normal_distribution(0, std);
		}
	}
}

void weight_initialization_uniform(double** const out, uint64_t in_size, uint64_t out_size, double a, double b){
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++i){
			out[i][k] = uniform_distribution(a, b);
		}
	}
}

void weight_initialization_normal(double** const out, uint64_t in_size, uint64_t out_size, double a, double b){
	for (uint64_t i = 0;i<out_size;++i){
		for (uint64_t k = 0;k<in_size;++i){
			out[i][k] = normal_distribution(a, b);
		}
	}
}

void loss_mse_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = 2*(result[i]-expected[i]);
	}
}

void loss_mae_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double term = result[i]-expected[i];
		output[i] = term/fabsf(term);
	}
}

void loss_mape_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		output[i] = (1/powf(expected[i], 2))*term/fabsf(term);
	}
}

void loss_huber_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double term = expected[i]-result[i];
		if (term <= a){
			output[i] = term;
			continue;
		}
		output[i] = a*(int)((0<term)-(term<0));
	}
}

void loss_huber_modified_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
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

void loss_cross_entropy_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = (-(expected[i]/result[i]))+((1-expected[i])/(1-result[i]));
	}
}

void loss_hinge_partial(double* const output, const double* const result, const double* const expected, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (expected[i]*result[i] >= 1){
			output[i] = 0;
		}
		output[i] = -expected[i];
	}
}

void activation_sigmoid_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		float fx = 1/(1+expf(-buffer[i]));
		output[i] = fx*(1-fx);
	}
}

void activation_relu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = (buffer[i] > 0);
	}
}

void activation_tanh_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		output[i] = 1-powf(tanh(buffer[i]), 2);
	}
}

void activation_linear_partial(double* const output, const double* const buffer, uint64_t size, double a){
	memset(output, 1, size*sizeof(float));
}

void activation_relu_leaky_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = 0.01;
	}
}

void activation_relu_parametric_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a;
	}
}

void activation_elu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		if (buffer[i] > 0){
			output[i] = 1;
			continue;
		}
		output[i] = a+(a*(expf(buffer[i])-1));
	}
}

void activation_softmax_partial(double* const output, const double* const buffer, uint64_t size, double a){
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

void activation_swish_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		float fx = 1/(1+expf(-buffer[i]*a));
		output[i] = fx+a*buffer[i]*fx*(1-fx);
	}
}

void activation_gelu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	float s2op = sqrt(2/M_PI);
	for (uint64_t i = 0;i<size;++i){
		float x = buffer[i];
		float inside = tanh(s2op*(x+GELU_C*powf(x, 3)));
		output[i] = 0.5*(1+inside)+0.5*x*(1-powf(inside, 2))*s2op*(1+3*GELU_C*x*x);
	}
}

void activation_selu_partial(double* const output, const double* const buffer, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			output[i] = SELU_LAMBDA;
			continue;
		}
		output[i] = SELU_LAMBDA*SELU_ALPHA*expf(x);
	}
}

double loss_mse(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += loss*loss;
	}
	return (sum)/(size);
}

double loss_mae(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float loss = expected[i]-result[i];
		buffer[i] = loss;
		sum += abs(loss);
	}
	return sum/(size);
}

double loss_mape(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float loss = expect-result[i];
		buffer[i] = loss;
		sum += abs(loss/expect);
	}
	return sum/(size);
}

double loss_huber(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
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

double loss_huber_modified(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
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

double loss_cross_entropy(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += expect*log(res);
	}
	return -sum;
}

double loss_hinge(double* const buffer, const double* const result, const double* const expected, uint64_t size, double a){
	float sum = 0;
	for (uint64_t i = 0;i<size;++i){
		float expect = expected[i];
		float res = result[i];
		buffer[i] = expect-res;
		sum += fmax(0,1-(expect*res));
	}
	return sum;
}

void activation_sigmoid(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = 1/(1+expf(-output[i]));
	}
}

void activation_relu(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = fmax(0,output[i]);
	}
}

void activation_tanh(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = tanh(output[i]);
	}
}

void activation_binary_step(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = output[i] >= 0;
	}
}

void activation_linear(double* const buffer, const double* const output, uint64_t size, double a){
	memcpy(buffer, output, size*sizeof(double));
}

void activation_relu_leaky(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = fmax(0.1*x,x);
	}
}

void activation_relu_parametric(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = fmax(a*x, x);
	}
}

void activation_elu(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		if (x < 0){
			buffer[i] = a*(expf(x)-1);
		}
	}
}

void activation_softmax(double* const buffer, const double* const output, uint64_t size, double a){
	float denom = 0;
	for (uint64_t i = 0;i<size;++i){
		denom += expf(output[i]);
	}
	for (uint64_t i = 0;i<size;++i){
		buffer[i] = expf(output[i])/denom;
	}
}

void activation_swish(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = x/(1+expf(-x));
	}
}

void activation_gelu(double* const buffer, const double* const output, uint64_t size, double a){
	float x;
	const float s2p  = sqrtf(2/M_PI);
	for (uint64_t i = 0;i<size;++i){
		x = output[i];
		buffer[i] = 0.5*x*(1+tanh(s2p*(x+(GELU_C*pow(x,3)))));
	}
}

void activation_selu(double* const buffer, const double* const output, uint64_t size, double a){
	for (uint64_t i = 0;i<size;++i){
		double x = output[i];
		if (x > 0){
			buffer[i] = SELU_LAMBDA * x;
			continue;
		}
		buffer[i] = SELU_LAMBDA*SELU_ALPHA*(expf(x)-1);
	}
}

int main(int argc, char** argv){
	return 0;
}
