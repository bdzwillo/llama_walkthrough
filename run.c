/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------
// Tokenizer

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct Tokenizer {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Tokenizer *tokenizer;
    int embedding;
    int added; // show added weights
    int attention;
    int layer; // layer to tyrace
    int head; // head to trace
} Trace;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
    Trace *trace; // enable trace output
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
    t->trace = NULL;
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, const float* x, const float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void layernorm(float *o, const float *x, const float* weight, int size) {
    // calculate mean and variance
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int j = 0; j < size; j++) {
        sum += x[j];
        sum_sq += x[j] * x[j];
    }
    float mean = sum / size;
    float variance = (sum_sq / size) - (mean * mean);
    float eps = 1e-5f;

    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = (x[j] - mean) / sqrtf(variance + eps);
        o[j] = weight[j] * o[j];
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void mattranspose(float* xout, float* x, int n, int d) {
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < n; j++) {
            xout[j * d + i] = x[i * n + j];
	}
    }
}

int topk(const float *x, int dim, int k, int *top)
{
	if (k > dim) {
		return -1;
	}
	memset(top, 0, k * sizeof(int));

	for (int i=0; i < dim; i++) {
		if (x[top[k-1]] <= x[i]) {
			top[k-1] = i;

			// sort index into top[k] array
			for (int j=k-1; j > 0; j--) {
				if (x[top[j]] < x[top[j-1]]) {
					break;
				}
				int tmp = top[j];
				top[j] = top[j-1];
				top[j-1] = tmp;
			}
		}
	}
	return 0;
}

int sum(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += v[i];
    }
    return sum;
}

float mean(const float *v, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += v[i];
    }
    return (float)(sum / n);
}

float variance(const float *v, int n) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += v[j];
        sum_sq += v[j] * v[j];
    }
    float mean = sum / n;
    return (float)((sum_sq / n) - (mean * mean));
}

// ----------------------------------------------------------------------------
// helper

void vec_copy(float *xout, const float *x, int size) {
	memcpy(xout, x, size * sizeof(float));
}

void vec_dump(const float *x, int dim, int win)
{
	int i;

	printf("[");
	for (i=0; i < dim; i++) {
		if (i >= win) {
			break;
		}
		if (i > 0) {
			printf(", ");
		}
		printf("%7.4f", x[i]);
	}
	if (i < (dim-win)) {
		i = dim-win;
		printf(", ...");
	}
	for (; i < dim; i++) {
		printf(", %7.4f", x[i]);
	}
	printf("]");
}

void vec_top_dump(const float *x, int dim, int win)
{
	int top[win];

	topk(x, dim, win, top);

	printf("[");
	for (int i=0; i < win; i++) {
		if (i > 0) {
			printf(", ");
		}
		printf("%d: %1.4f", top[i], x[top[i]]);
	}
	printf("]");
}

void safe_printf(const char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

void output_ids(const int *token, int n) {
    printf("[");
    for (int i=0; i < n; i++) {
	if (i > 0) {
		printf(", ");
	}
	printf("%d", token[i]);
    }
    printf("]");
    printf("\n");
}

void output_tokens(char **vocab, const int *token, int n) {
    printf("[");
    for (int i=0; i < n; i++) {
	if (i > 0) {
		printf(", ");
	}
	//printf("'%s'", vocab[token[i]]);
	printf("'");
	safe_printf(vocab[token[i]]);
	printf("'");
    }
    printf("]");
    printf("\n");
}

int output_topk_with_index(Tokenizer *tokenizer, const float *logits, int n, int with_index) {
    if (n > tokenizer->vocab_size) {
        n = tokenizer->vocab_size;
    }
    int toklen = 0;
    int top[n];
    float p_logits[tokenizer->vocab_size];
    vec_copy(p_logits, logits, tokenizer->vocab_size);
    softmax(p_logits, tokenizer->vocab_size); // convert to propabilities
    topk(p_logits, tokenizer->vocab_size, n, top);

    if (with_index) {
        printf("{");
    } else {
        printf("[");
    }
    for (int i=0; i < n; i++) {
	char *piece = tokenizer->vocab[top[i]];
	int percent = (int)(p_logits[top[i]] * 100);
	if (i > 0) { printf(", "); }
        if (with_index) {
	    printf("%d: ('%s', %d)", top[i], piece, percent);
        } else {
	    printf("('%s', %d)", piece, percent);
        }
        toklen += strlen(piece); if (percent > 9) { toklen++; }
    }
    if (with_index) {
        printf("}");
    } else {
        printf("]");
    }
    return toklen;
}

int output_topk(Tokenizer *tokenizer, const float *logits) {
    return output_topk_with_index(tokenizer, logits, 5, 0);
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    if (transformer->trace && transformer->trace->embedding) { // output embedding layer
        rmsnorm(s->xb, x, w->rms_final_weight, dim);
        matmul(s->logits, s->xb, w->wcls, p->dim, p->vocab_size);
        printf(" E:"); output_topk(transformer->trace->tokenizer, s->logits);
        printf("\n");
    }

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        if (transformer->trace && transformer->trace->attention) { // output single head attention mask
            if (l == transformer->trace->layer) {
                int h = transformer->trace->head;
                printf("%12s:", transformer->trace->tokenizer->vocab[token]);
                printf(" A[%llu,%d]:", l, h); vec_dump(s->att + h * p->seq_len, pos+1, 8); printf("\n");
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }

        if (transformer->trace && transformer->trace->embedding) { // output layer
            float xtmp[dim];
            rmsnorm(xtmp, x, w->rms_final_weight, dim);
            matmul(s->logits, xtmp, w->wcls, p->dim, p->vocab_size);
            printf("%2lld:", l);
            int maxoff = 40;
            int toklen = output_topk(transformer->trace->tokenizer, s->logits);
            if (transformer->trace->added) {
                for (int i = toklen; i < maxoff; i++) {
                    printf(" ");
                }
                printf(" ");
                rmsnorm(xtmp, s->xb, w->rms_final_weight, dim);
                matmul(s->logits, xtmp, w->wcls, p->dim, p->vocab_size);
                printf("ADDED:");
                output_topk(transformer->trace->tokenizer, s->logits);
            }
            printf("\n");
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
        // remove tokenizer.bin changes again.
        if (i==1) { snprintf(t->vocab[i], len+1, "<s>"); }
        if (i==2) { snprintf(t->vocab[i], len+1, "<\\s>"); }
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(const float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, const float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        float p_logits[sampler->vocab_size];

        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { p_logits[q] = logits[q] / sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(p_logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(p_logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(p_logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

int get_tokens(Tokenizer *tokenizer, char *prompt, int *tokens, int max_tokens) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }
    if (num_prompt_tokens > max_tokens) {
        num_prompt_tokens = max_tokens;
    }
    memcpy(tokens, prompt_tokens, max_tokens * sizeof(int));
    free(prompt_tokens);
    return num_prompt_tokens;
}

void output_string(Tokenizer *tokenizer, int prev_token, int token) {
        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, prev_token, token);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
}

void output_formatted_token(Tokenizer *tokenizer, int token) {
        char cur[64];
        snprintf(cur, sizeof(cur), "'%s'", tokenizer->vocab[token]);
        printf("%12s:", cur);
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    int max_tokens = strlen(prompt)+3; // +3 for '\0', ?BOS, ?EOS
    int prompt_tokens[max_tokens];
    int num_prompt_tokens = get_tokens(tokenizer, prompt, prompt_tokens, max_tokens);

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        output_string(tokenizer, token, next);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
}

void generate_greedy(Transformer *transformer, Tokenizer *tokenizer, char *prompt, int steps) {
    int token[steps+1];
    int num_prompt_tokens = get_tokens(tokenizer, prompt, token, steps);
    int pos = 0;

    while (pos < steps) {
        float *logits = forward(transformer, token[pos], pos);
        pos++;

        if (pos >= num_prompt_tokens) {
            // when prompt is completed, append token with the highest probability
            token[pos] = sample_argmax(logits, tokenizer->vocab_size);

            if (token[pos] == 1) { break; } // stop at BOS==1 token
        }
        output_string(tokenizer, token[pos-1], token[pos]);
    }
    printf("\n");
}

void generate_topk(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    int token[steps+1];
    int num_prompt_tokens = get_tokens(tokenizer, prompt, token, steps);
    int pos = 0;

    while (pos < steps) {
        float *logits = forward(transformer, token[pos], pos);
        pos++;

        if (pos >= num_prompt_tokens) {
            // when prompt is completed, append token with the highest probability
            token[pos] = sample(sampler, logits);

            if (token[pos] == 1) { break; } // stop at BOS==1 token
        }
        output_formatted_token(tokenizer, token[pos]);
        output_topk(tokenizer, logits);
        printf("\n");
    }
}

void generate_layers(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    int token[steps+1];
    int num_prompt_tokens = get_tokens(tokenizer, prompt, token, steps);
    int pos = 0;

    Trace trace = { .tokenizer = tokenizer, .embedding = 1, .added = 1 };
    transformer->trace = &trace;

    while (pos < steps) {
        float *logits = forward(transformer, token[pos], pos);
        pos++;

        if (pos >= num_prompt_tokens) {
            // when prompt is completed, append token with the highest probability
            token[pos] = sample(sampler, logits);

            if (token[pos] == 1) { break; } // stop at BOS==1 token
        }
        output_formatted_token(tokenizer, token[pos]);
        output_topk(tokenizer, logits);
        printf("\n");
    }
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx = 0;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next = 0;    // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    // int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// tokenize test

void tokenize(Tokenizer *tokenizer, char *prompt) {
    int max_tokens = strlen(prompt)+3; // +3 for '\0', ?BOS, ?EOS
    int token[max_tokens];
    int num_tokens = get_tokens(tokenizer, prompt, token, max_tokens);

    output_ids(token, num_tokens);
    output_tokens(tokenizer->vocab, token, num_tokens);
}

// ----------------------------------------------------------------------------
// embedding test

void embed(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    int max_tokens = prompt ? strlen(prompt)+3 : 1; // +3 for '\0', ?BOS, ?EOS
    int prompt_tokens[max_tokens];
    int num_prompt_tokens = get_tokens(tokenizer, prompt, prompt_tokens, max_tokens);

    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    float embeds[p->dim];

    // dump whole vacabulary if prompt is empty
    for (int pos=0; pos < (prompt ? num_prompt_tokens : p->vocab_size); pos++) {
        int token = prompt ? prompt_tokens[pos] : pos;
        // copy the token embedding
        vec_copy(embeds, w->token_embedding_table + token * p->dim, p->dim);

        float w_sum = sum(embeds, p->dim);
        output_formatted_token(tokenizer, token);
        vec_dump(embeds, p->dim, 3);
        printf(", SUM: %3.2f, TOP:", w_sum);
        vec_top_dump(embeds, p->dim, 3);
        printf("\n");
    }
}

void embed_tokens(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    int max_tokens = prompt ? strlen(prompt)+3 : 1; // +3 for '\0', ?BOS, ?EOS
    int prompt_tokens[max_tokens];
    int num_prompt_tokens = get_tokens(tokenizer, prompt, prompt_tokens, max_tokens);

    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    float embeds[p->dim];
    float logits[p->vocab_size];

    // dump whole vacabulary if prompt is empty
    for (int pos=0; pos < (prompt ? num_prompt_tokens : p->vocab_size); pos++) {
        int token = prompt ? prompt_tokens[pos] : pos;
        // copy the token embedding
        vec_copy(embeds, w->token_embedding_table + token * p->dim, p->dim);

        rmsnorm(embeds, embeds, w->rms_final_weight, p->dim);
        matmul(logits, embeds, w->wcls, p->dim, p->vocab_size);

        output_formatted_token(tokenizer, token);
        output_topk_with_index(tokenizer, logits, 5, 1);
        printf("\n");
    }
}

// ----------------------------------------------------------------------------
// distance test

int getnext(Transformer *t, Tokenizer *tokenizer, char *prompt, float *weights) {
    int max_tokens = strlen(prompt)+3; // +3 for '\0', ?BOS, ?EOS
    int prompt_tokens[max_tokens];
    int num_prompt_tokens = get_tokens(tokenizer, prompt, prompt_tokens, max_tokens);

    // reinit the RunState buffers
    free_run_state(&t->state);
    malloc_run_state(&t->state, &t->config);

    int pos = 0;     // position in the sequence
    int token;
    float *logits;
    do {
        token = prompt_tokens[pos];
        logits = forward(t, token, pos);
        pos++;
    } while (pos < num_prompt_tokens);
    token = sample_argmax(logits, tokenizer->vocab_size);

    output_formatted_token(tokenizer, token);
    vec_top_dump(t->state.x, t->config.dim, 3);
    printf(", TOP:");
    output_topk_with_index(tokenizer, logits, 5, 0);
    printf("\n");

    memcpy(weights, t->state.x, t->config.dim*sizeof(float));
    return token;
}

// 1.0 = similar
// (cosine_similarity is identical to dot-product on input vectors with mean=0 and variance=1)
//
float cosine_similarity(const float* v1, const float* v2, int n) {
    double dot = 0.0;
    double mag1 = 0.0;
    double mag2 = 0.0;

    // calc dot product & magnitutes of each vector
    for (int i = 0; i < n; i++) {
        dot += v1[i] * v2[i];
        mag1 += v1[i] * v1[i];
        mag2 += v2[i] * v2[i];
    }
    mag1 = sqrt(mag1);
    mag2 = sqrt(mag2);

    if (mag1 == 0.0 || mag2 == 0.0) {
        return 0.0;
    }
    return (float)(dot / (mag1 * mag2));
}

void distance(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;

    // the embedding vectors predict the value of the next token
    //
    char *q_man = "An adult boy is a";
    char *q_woman = "An adult girl is a";
    char *q_king = "Long live the";
    char *q_queen = "God save the";

    if (prompt) {
        char *e;
        if ((e = strtok(prompt, "/"))) { q_man = e; }
        if ((e = strtok(NULL, "/"))) { q_woman = e; }
        if ((e = strtok(NULL, "/"))) { q_king = e; }
        if ((e = strtok(NULL, "/"))) { q_queen = e; }
    }

    printf("Generate word embeddings for (%s../%s../%s..):\n", q_man, q_woman, q_king);

    float man[p->dim];
    float woman[p->dim];
    float king[p->dim];
    float queen[p->dim];
    int t_man = getnext(transformer, tokenizer, q_man, man);
    int t_woman = getnext(transformer, tokenizer, q_woman, woman);
    int t_king = getnext(transformer, tokenizer, q_king, king);
    int t_queen = getnext(transformer, tokenizer, q_queen, queen);
    char *s_man = tokenizer->vocab[t_man];
    char *s_woman = tokenizer->vocab[t_woman];
    char *s_king = tokenizer->vocab[t_king];
    char *s_queen = tokenizer->vocab[t_queen];

    printf("\n");
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_man,   s_man,   cosine_similarity(man, man, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_man,   s_woman, cosine_similarity(man, woman, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_man,   s_king,  cosine_similarity(man, king, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_woman, s_king,  cosine_similarity(woman, king, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_man,   s_queen, cosine_similarity(man, queen, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_woman, s_queen, cosine_similarity(woman, queen, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_king,  s_queen, cosine_similarity(king, queen, p->dim));
    printf("\n");

    float x[p->dim];

    // test example: “king” - “man” + “woman” =~ "queen"
    //
    for (int i = 0; i < p->dim; i++) {
        x[i] = king[i] - man[i] + woman[i];
    }
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);

    int token = sample_argmax(s->logits, tokenizer->vocab_size);

    printf("Calculate '%s' - '%s' + '%s' =~ '%s':\n", s_man, s_woman, s_king, tokenizer->vocab[token]);

    output_formatted_token(tokenizer, token);
    vec_top_dump(x, p->dim, 3);
    printf(", TOP:");
    output_topk_with_index(tokenizer, s->logits, 5, 0);
    printf("\n");

    printf("\n");
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_man,   tokenizer->vocab[token], cosine_similarity(man, x, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_woman, tokenizer->vocab[token], cosine_similarity(woman, x, p->dim));
    printf("  similarity(%-6s,%-6s): %1.8f\n", s_king,  tokenizer->vocab[token], cosine_similarity(king, x, p->dim));
}

// ----------------------------------------------------------------------------
// position test

void position(Transformer *transformer, int steps) {
    Config* p = &transformer->config;
    float vc[p->dim/2];
    float vs[p->dim/2];

    for (int pos=0; pos < steps; pos++) {
        for (int i = 0; i < p->dim; i+=2) {
            float freq = 1.0f / powf(10000.0f, i / (float)p->dim);
            float val = pos * freq;

            vc[i/2] = cosf(val);
            vs[i/2] = sinf(val);
        }
        printf("%2d: cos:", pos); vec_dump(vc, p->dim/2, 6);
        printf("\n");
        printf("%2d: sin:", pos); vec_dump(vs, p->dim/2, 6);
        printf("\n");
    }
}

void position_rope(Transformer *transformer, float xval, float yval, int steps, int do_rope) {
    Config* p = &transformer->config;
    float q[p->dim];

    for (int pos=0; pos < steps; pos++) {
        // initialize test query with repeated (x, y) tupels
        for (int i = 0; i < p->dim; i+=2) {
            q[i]   = xval;
            q[i+1] = yval;
        }
        for (int i = 0; i < p->dim; i+=2) {
            float freq = 1.0f / powf(10000.0f, i / (float)p->dim);
            float val = pos * freq;

            float v0 = q[i];
            float v1 = q[i+1];
            if (do_rope) {
                q[i]   = v0 * cosf(val) - v1 * sinf(val);
                q[i+1] = v0 * sinf(val) + v1 * cosf(val);
            } else {
                q[i]   = v0 + cosf(val);
                q[i+1] = v1 + sinf(val);
            }
        }
        printf("%2d:", pos); vec_dump(q, p->dim, 8);
        printf("\n");
    }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|generate_greedy|generate_topk|generate_layers|tokenize|emded|embed_tokens|distance|position|position_rope, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // ovrerride to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else if (strcmp(mode, "generate_greedy") == 0) {
        generate_greedy(&transformer, &tokenizer, prompt, steps);
    } else if (strcmp(mode, "generate_topk") == 0) {
        generate_topk(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "generate_layers") == 0) {
        generate_layers(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "tokenize") == 0) {
        tokenize(&tokenizer, prompt);
    } else if (strcmp(mode, "embed") == 0) {
        embed(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "embed_tokens") == 0) {
        embed_tokens(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "distance") == 0) {
        distance(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "position") == 0) {
        position(&transformer, steps);
    } else if (strcmp(mode, "position_rope") == 0) {
        position_rope(&transformer, 1.0f, 1.0f, steps, 1);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
