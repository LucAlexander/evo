#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

#include "kickstart.h"

pool pool_alloc(uint64_t cap, POOL_TAG t){
	void* mem = malloc(cap);
	if (mem == NULL || t == NO_POOL){
		return (pool){.tag=NO_POOL};
	}
	return (pool){
		.tag = t,
		.buffer = mem,
		.ptr = mem,
		.left = cap,
		.next = NULL
	};
}

void pool_empty(pool* const p){
	p->left += p->ptr - p->buffer;
	p->ptr = p->buffer;
	if (p->next != NULL){
		pool_empty(p->next);
	}
}

void pool_dealloc(pool* const p){
	free(p->buffer);
	if (p->next != NULL){
			pool_dealloc(p->next);
	}
	free(p->next);
}

void* pool_request(pool* const p, uint64_t bytes){
	if (p->left < bytes){
		uint64_t capacity = p->left + (p->ptr-p->buffer);
		if (p->tag == POOL_STATIC || bytes > capacity){
			return NULL;
		}
		if (p->next == NULL){
			p->next = malloc(sizeof(pool));
			*p->next = pool_alloc(p->left + (p->ptr-p->buffer), POOL_DYNAMIC);
		}
		return pool_request(p->next, bytes);
	}
	p->left -= bytes;
	void* addr = p->ptr;
	p->ptr += bytes;
	return addr;
}

void* pool_request_aligned(pool* const p, uint64_t bytes, uint16_t alignment){
	uint64_t original = bytes;
	uint64_t diff = alignment-((uint64_t)(p->ptr) % alignment);
	if (diff != 0){
		bytes += diff;
	}
	if (p->left < bytes){
		uint64_t capacity = p->left + (p->ptr-p->buffer);
		if (p->tag == POOL_STATIC || bytes > capacity){
			return NULL;
		}
		if (p->next == NULL){
			p->next = malloc(sizeof(pool));
			*p->next = pool_alloc(p->left + (p->ptr-p->buffer), POOL_DYNAMIC);
		}
		return pool_request_aligned(p->next, original, alignment);
	}
	p->left -= bytes;
	void* addr = p->ptr;
	p->ptr += bytes;
	addr += diff;
	return addr;
}

void* pool_byte(pool* const p){	
	if (p->left <= 0 || p->tag == POOL_DYNAMIC){
		return NULL;
	}
	p->left -= 1;
	void* addr = p->ptr;
	p->ptr += 1;
	return addr;
}

void pool_save(pool* const p){
	if (p->next != NULL){
		pool_save(p->next);
		return;
	}
	p->ptr_save = p->ptr;
	p->left_save = p->left;
}

void pool_load(pool* const p){
	if (p->next != NULL){
		pool_load(p);
		return;
	}
	p->ptr = p->ptr_save;
	p->left = p->left_save;
}

string string_copy(pool* const mem, string* const src){
	string new = {
		.len = src->len
	};
	new.str = pool_request(mem, src->len);
	for (uint64_t i = 0;i<src->len;++i){
		new.str[i] = src->str[i];
	}
	return new;
}

string string_init(pool* const mem, char* src){
	string new = {
		.str = pool_request(mem, 1),
		.len = 0
	};
	char* c = src;
	for (uint64_t i = 0;i<ITERATION_LIMIT_MAX;++i){
		if (*c == '\0'){
			break;
		}
		pool_request(mem, 1);
		new.str[new.len] = *c;
		new.len += 1;
		c += 1;
	}
	return new;
}

void string_set(pool* const mem, string* const str, char* src){
	uint64_t new_len = 0;
	char* c = src;
	for (uint64_t i = 0;i<ITERATION_LIMIT_MAX;++i){
		if (*c == '\0'){
			break;
		}
		str->str[new_len] = src[new_len];	
		new_len += 1;
		if (new_len > str->len){
			string new = string_init(mem, src);
			str->str = new.str;
			str->len = new.len;
			return;
		}
	}
	str->len = new_len;
}

void string_print(string* const str){
	for (uint64_t i = 0;i<str->len;++i){
		putchar(str->str[i]);
	}
}

void string_cat(pool* const mem, string* const a, string* const b){
	uint64_t len = a->len + b->len;
	char* new = pool_request(mem, len);
	uint64_t i = 0;
	for (;i<a->len;++i){
		new[i] = a->str[i];
	}
	for (;i<len;++i){
		new[i] = b->str[i-a->len];
	}
	a->str = new;
	a->len = len;
}
