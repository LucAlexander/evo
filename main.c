#include "kickstart.h"

network
network_init(pool* const mem, uint64_t input, uint64_t output){
	network net = {
		.mem = mem,
		.input = input_init(mem, input),
		.output = layer_init(mem, output),
		.loss = NULL,
		.bias = NULL,
		.weight = NULL
	};
}

layer*
input_init(pool* const mem, uint64_t width){
	layer* input = pool_request(mem, sizeof(layer));
	input->tag = INPUT_NODE;
	input->data.input.output = pool_request(mem, sizeof(double)*width);
	input->data.input.width = width;
	input->prev = NULL;
	input->prev_count = 0;
	input->prev_capacity = 0
	input->next = pool_request(mem, 2*sizeof(layer*));
	input->next_count = 0;
	input->next_capacity = 2
	return input;
}

layer*
layer_init(pool* const mem, uint64_t width){
	layer* node = pool_request(mem, sizeof(layer));
	node->tag = LAYER_NODE;
	node->data.layer.output = pool_request(mem, sizeof(double)*width);
	node->data.layer.width = width;
	node->data.layer.weights = pool_request(mem, sizeof(double*)*width);
	node->data.layer.bias = pool_request(mem, sizeof(double*)*width);
	node->data.layer.activation = NULL;
	node->prev = pool_request(mem, 2*sizeof(layer*));
	node->prev_count = 0;
	node->prev_capacity = 2;
	node->next = pool_request(mem, 2*sizeof(layer*));
	node->next_count = 0;
	node->next_capacity = 2;
	return node;
}

void
layer_link(pool* const mem, layer* const a, layer* const b){
	if (b->tag == INPUT_NODE){
		fprintf(stderr, "tried to prepend input\n");
		return NULL;
	}
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
	if (b->tag == INPUT_NODE){
		fprintf(stderr, "Tried to unlink prepension to input\n");
		return;
	}
	uint8_t found = 0;
	for (uint64_t i = 0;i<a->next_count;++i){
		if (a->next[i] == b){
			found = 1;
		}
		else if (found == 1){
			a->next[i-1] = a->next[i];
		}
	}
	if (found == 1){
		a->next_count -= 1;
		found = 0;
	}
	else{
		fprintf(stderr, "Couldnt unlink, not found in next\n");
		return;
	}
	for (uint64_t i = 0;i<b->prev_count;++i){
		if (b->prev[i] == a){
			found = 1;
		}
		else if (found == 1){
			b->prev[i-1] = b->prev[i];
		}
	}
	if (found == 1){
		b->prev_count -= 1;
	}
	else{
		fprintf(stderr, "Couldnt unlink, not found in prev\n");
	}
}

void
layer_insert(pool* const mem, layer* const a, layer* const b, layer* const c){
	layer_unlink(a, b);
	layer_link(mem, a, b);
	layer_link(mem, b, c);
}

void
forward(layer* const node){
	if (node->tag == INPUT_NODE){
		for (uint64_t i = 0;i<node->next_count;++i){
			forward(node->next[i]);
		}
		return;
	}
	//TODO forward pass code
	for (uint64_t i = 0;i<node->next_count;++i){
		forward(node->next[i]);
	}
}

void backward(layer* const node){
	//TODO
}

int main(int argc, char** argv){
	return 0;
}
