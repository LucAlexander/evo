#ifndef KICKSTART_H
#define KICKSTART_H

#include <inttypes.h>

#define ITERATION_LIMIT_MAX 65536
#define ITERATION_LIMIT_SOFT 2048
#define ITERATION_LIMIT_HARD 256

typedef enum POOL_TAG {
	POOL_STATIC=0,
	POOL_DYNAMIC=1,
	NO_POOL
} POOL_TAG;

typedef struct pool {
	POOL_TAG tag;
	void* buffer;
	void* ptr;
	uint64_t left;
	struct pool* next;
	void* ptr_save;
	uint64_t left_save;
} pool;

pool pool_alloc(uint64_t cap, POOL_TAG t);
void pool_empty(pool* const p);
void pool_dealloc(pool* const p);
void* pool_request(pool* const p, uint64_t bytes);
void* pool_request_aligned(pool* const p, uint64_t bytes, uint16_t alignment);
void* pool_byte(pool* const p);
void pool_save(pool* const p);
void pool_load(pool* const p);

typedef struct string {
	char* str;
	uint64_t len;
} string;

string string_init(pool* const mem, char* src);
string string_copy(pool* const mem, string* const src);
void string_set(pool* const mem, string* const str, char* src);
void string_print(string* const str);
void string_cat(pool* const mem, string* const a, string* const b);

#define MAP_BUCKET_COUNT 128
void string_print(string* const key);

typedef enum MAP_BUCKET_TAG {
	MAP_BUCKET_EMPTY,
	MAP_BUCKET_FULL
} MAP_BUCKET_TAG;

#define MAP_DECL(type)\
	typedef struct type##_map_bucket type##_map_bucket;\
	typedef struct type##_map type##_map;\
\
	typedef struct type##_map_bucket {\
		type data;\
		type##_map_bucket* left;\
		type##_map_bucket* right;\
		uint64_t hash;\
		string key;\
		MAP_BUCKET_TAG tag;\
	} type##_map_bucket;\
\
	uint8_t type##_map_bucket_insert(type##_map* map, type##_map_bucket* bucket, string* key, uint64_t hash, type value);\
	type* type##_map_bucket_access(type##_map* map, type##_map_bucket* bucket, string* key, uint64_t hash);\
\
	typedef struct type##_map {\
		type##_map_bucket buckets[MAP_BUCKET_COUNT];\
		pool* mem;\
	} type##_map;\
\
	type##_map type##_map_init(pool* mem);\
	uint8_t type##_map_insert(type##_map* map, string key, type value);\
	type* type##_map_access(type##_map* map, string key);\
	void type##_map_clear(type##_map* map);

#define MAP_IMPL(type)\
	uint8_t type##_map_bucket_insert(type##_map* map, type##_map_bucket* bucket, string* key, uint64_t hash, type value){\
		if (bucket->tag == MAP_BUCKET_EMPTY){\
			bucket->data = value;\
			bucket->hash = hash;\
			bucket->key = string_copy(map->mem, key);\
			bucket->tag = MAP_BUCKET_FULL;\
			bucket->left = pool_request(map->mem, sizeof(type##_map_bucket));\
			bucket->right = pool_request(map->mem, sizeof(type##_map_bucket));\
			bucket->left->tag = MAP_BUCKET_EMPTY;\
			bucket->right->tag = MAP_BUCKET_EMPTY;\
			return 0;\
		}\
		if (hash < bucket->hash){\
			return type##_map_bucket_insert(map, bucket->left, key, hash, value);\
		}\
		else if (hash > bucket->hash){\
			return type##_map_bucket_insert(map, bucket->right, key, hash, value);\
		}\
		bucket->data = value;\
		return 1;\
	}\
\
	type* type##_map_bucket_access(type##_map* map, type##_map_bucket* bucket, string* key, uint64_t hash){\
		if (bucket->tag == MAP_BUCKET_EMPTY){\
			return NULL;\
		}\
		if (hash < bucket->hash){\
			return type##_map_bucket_access(map, bucket->left, key, hash);\
		}\
		if (hash > bucket->hash){\
			return type##_map_bucket_access(map, bucket->left, key, hash);\
		}\
		return &bucket->data;\
	}\
\
	type##_map type##_map_init(pool* mem){\
		type##_map map = {\
			.mem=mem\
		};\
		for (uint64_t i = 0;i<MAP_BUCKET_COUNT;++i){\
			map.buckets[i].tag = MAP_BUCKET_EMPTY;\
		}\
		return map;\
	}\
\
	uint8_t type##_map_insert(type##_map* map, string key, type value){\
		uint32_t hash = 5381;\
		int16_t c;\
		for (uint64_t i = 0;i<key.len;++i){\
			c = key.str[i];\
			hash = ((hash<<5)+hash)+c;\
		}\
		uint64_t bucket = hash % MAP_BUCKET_COUNT;\
		return type##_map_bucket_insert(map, &map->buckets[bucket], &key, hash, value);\
	}\
\
	type* type##_map_access(type##_map* map, string key){\
		uint32_t hash = 5381;\
		int16_t c;\
		for (uint64_t i = 0;i<key.len;++i){\
			c = key.str[i];\
			hash = ((hash<<5)+hash)+c;\
		}\
		uint64_t bucket = hash % MAP_BUCKET_COUNT;\
		return type##_map_bucket_access(map, &map->buckets[bucket], &key, hash);\
	}\
\
	void type##_map_clear(type##_map* map){\
		for (uint64_t i = 0;i<MAP_BUCKET_COUNT;++i){\
			map->buckets[i].tag = MAP_BUCKET_EMPTY;\
		}\
	}

#endif
