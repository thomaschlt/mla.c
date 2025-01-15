#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

// forward declarations
typedef struct MLA_Layer
{
    int d;                // input dimension
    int n_h;              // number of heads
    int qk_nope_head_dim; // query/key non-RoPE dimension per head
    int qk_rope_head_dim; // query/key RoPE dimension per head
    int d_c;              // KV compression dimension
    int d_c_prime;        // query compression dimension
    int v_head_dim;       // value dimension
    float **W_DQ;         // down-projection matrice for queries
    float **W_UQ_nope;    // up-projection matrice for non-RoPE queries (qk_nope_head_dim x d_c_prime)
    float **W_UQ_rope;    // up-projection matrice for RoPE queries (qk_rope_head_dim x d_c_prime)
    float **W_UK_nope;    // up-projection matrice for non-RoPE keys (qk_nope_head_dim x d_c)
    float **W_UK_rope;    // up-projection matrice for RoPE keys (qk_rope_head_dim x d_c)
    float **W_UV;         // up-projection matrice for values (v_head_dim x d_c_prime)
    float **W_O;          // output projection matrice
    float **W_DKV;        // down-projection matrice for keys and values

    float ***K_cache;       // [seq_len][n_h][qk_head_dim]
    float ***V_cache;       // [seq_len][n_h][v_head_dim]
    int seq_len;            // current sequence length
    int max_seq_len;        // Maximum sequence length
    float *RMS_norm_weight; // RMSNorm weights
    float *norm_weight;     // RMSNorm weights
} MLA_Layer;

// complex number structure
typedef struct
{
    float real;
    float imag;
} Complex;

void init_random_matrice(float **matrice, int rows, int cols);
void init_random_vector(float *vec, int size);
void print_vector(const char *name, float *vec, int size);
MLA_Layer *init_mla_layer(int d, int n_h, int qk_nope_head_dim, int qk_rope_head_dim, int d_c, int d_c_prime, int v_head_dim, int max_seq_len);
void free_mla_layer(MLA_Layer *layer);
float *mla_forward(MLA_Layer *layer, float *h_t);
float *matmul(float **A, float *x, int m, int n);
void apply_rope(float *vec, int position, int qk_rope_head_dim);
float **allocate_matrice(int rows, int cols);
float dot_product_scalar(float *vec1, float *vec2, int len);
void free_matrice(float **matrice, int rows);
float *concat_with_rope(float *vec1, float *vec2, int len1, int len2);

// Complex number operations
Complex complex_mul(Complex a, Complex b)
{
    return (Complex){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real};
}

Complex polar(float r, float theta)
{
    return (Complex){r * cos(theta), r * sin(theta)};
}

void init_random_matrice(float **matrice, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrice[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

// Helper function to initialize a vector with random values
void init_random_vector(float *vec, int size)
{
    for (int i = 0; i < size; i++)
    {
        vec[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

void print_vector(const char *name, float *vec, int size)
{
    printf("%s: [", name);
    for (int i = 0; i < size && i < 5; i++)
    {
        printf("%.4f", vec[i]);
        if (i < size - 1 && i < 4)
            printf(", ");
    }
    if (size > 5)
        printf(", ...");
    printf("]\n");
}

MLA_Layer *init_mla_layer(int d, int n_h, int qk_nope_head_dim, int qk_rope_head_dim, int d_c, int d_c_prime, int v_head_dim, int max_seq_len)
{
    MLA_Layer *layer = (MLA_Layer *)malloc(sizeof(MLA_Layer));
    if (!layer)
        return NULL;

    layer->d = d;
    layer->n_h = n_h;
    layer->qk_nope_head_dim = qk_nope_head_dim;
    layer->qk_rope_head_dim = qk_rope_head_dim;
    layer->d_c = d_c;
    layer->d_c_prime = d_c_prime;
    layer->v_head_dim = v_head_dim;
    // projection matrices
    layer->W_DQ = allocate_matrice(d_c_prime, d);
    layer->W_UQ_nope = allocate_matrice(qk_nope_head_dim * n_h, d_c_prime);
    layer->W_UQ_rope = allocate_matrice(qk_rope_head_dim * n_h, d_c_prime);
    layer->W_UK_nope = allocate_matrice(qk_nope_head_dim * n_h, d_c);
    layer->W_UK_rope = allocate_matrice(qk_rope_head_dim * n_h, d_c);
    layer->W_UV = allocate_matrice(n_h * v_head_dim, d_c_prime);
    layer->W_O = allocate_matrice(d, n_h * v_head_dim);
    layer->W_DKV = allocate_matrice(d_c, d);

    // init caches
    layer->max_seq_len = max_seq_len;
    layer->seq_len = 0;
    layer->K_cache = malloc(max_seq_len * sizeof(float **));
    layer->V_cache = malloc(max_seq_len * sizeof(float **));
    for (int i = 0; i < max_seq_len; i++)
    {
        layer->K_cache[i] = allocate_matrice(n_h, qk_nope_head_dim);
        layer->V_cache[i] = allocate_matrice(n_h, v_head_dim);
    }

    // initialization of normalization weights
    layer->norm_weight = malloc(d * sizeof(float));
    if (!layer->norm_weight)
    {
        free_mla_layer(layer);
        return NULL;
    }

    // initialization of norm weights to 1.0
    for (int i = 0; i < d; i++)
    {
        layer->norm_weight[i] = 1.0f;
    }

    return layer;
}

void free_mla_layer(MLA_Layer *layer)
{
    if (layer == NULL)
        return;

    // free all projection matrices
    free_matrice(layer->W_DQ, layer->d_c_prime);
    free_matrice(layer->W_UQ_nope, layer->qk_nope_head_dim * layer->n_h);
    free_matrice(layer->W_UQ_rope, layer->qk_rope_head_dim * layer->n_h);
    free_matrice(layer->W_UK_nope, layer->qk_nope_head_dim * layer->n_h);
    free_matrice(layer->W_UK_rope, layer->qk_rope_head_dim * layer->n_h);
    free_matrice(layer->W_UV, layer->n_h * layer->v_head_dim);
    free_matrice(layer->W_O, layer->d);
    free_matrice(layer->W_DKV, layer->d_c);
    // free layer itself
    free(layer->norm_weight);
    free(layer);
}

float *matmul(float **A, float *x, int m, int n)
{
    float *result = malloc(m * sizeof(float));
    if (!result)
        return NULL;

    for (int i = 0; i < m; i++)
    {
        result[i] = 0;
        for (int j = 0; j < n; j++)
        {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}

float **allocate_matrice(int rows, int cols)
{
    float **matrice = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        matrice[i] = (float *)malloc(cols * sizeof(float));
    }
    return matrice;
}

// RoPE with complex numbers
void apply_rope(float *vec, int position, int qk_head_dim)
{
    for (int i = 0; i < qk_head_dim; i += 2)
    {
        float theta = position / pow(10000.0f, (2.0f * i) / qk_head_dim);

        // convert pair of real numbers to complex
        Complex x = {vec[i], vec[i + 1]};
        Complex freq = polar(1.0f, theta);

        Complex result = complex_mul(x, freq);

        vec[i] = result.real;
        vec[i + 1] = result.imag;
    }
}

float *RMS_norm(float *vec, float *weight, int dim)
{
    float *output = malloc(dim * sizeof(float));
    if (!output)
        return NULL;

    // sum of squares
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
    {
        ss += vec[i] * vec[i];
    }

    // RMS
    float rms = 1.0f / sqrt(ss / dim + 1e-5);

    // normalize and scale
    for (int i = 0; i < dim; i++)
    {
        output[i] = vec[i] * rms * weight[i];
    }
    return output;
}

float dot_product_scalar(float *vec1, float *vec2, int len)
{
    float result = 0.0f;
    for (int i = 0; i < len; i++)
    {
        result += vec1[i] * vec2[i];
    }
    return result;
}

int main()
{
    // random seed init
    srand(time(NULL));

    // test parameters
    int d = 512;
    int n_h = 8;
    int qk_nope_head_dim = 64;
    int qk_rope_head_dim = 32;
    int d_c = 128;
    int d_c_prime = 128;
    int v_head_dim = 128;
    int max_seq_len = 4;

    // init MLA layer
    MLA_Layer *layer = init_mla_layer(d, n_h, qk_nope_head_dim, qk_rope_head_dim, d_c, d_c_prime, v_head_dim, max_seq_len);
    if (!layer)
    {
        printf("Failed to initialize MLA layer\n");
        return 1;
    }

    // init weights with random values
    init_random_matrice(layer->W_DQ, d_c_prime, d);
    init_random_matrice(layer->W_UQ_nope, qk_nope_head_dim * n_h, d_c_prime);
    init_random_matrice(layer->W_UQ_rope, qk_rope_head_dim * n_h, d_c_prime);
    init_random_matrice(layer->W_UK_nope, qk_nope_head_dim * n_h, d_c);
    init_random_matrice(layer->W_UK_rope, qk_rope_head_dim * n_h, d_c);
    init_random_matrice(layer->W_UV, layer->n_h * layer->v_head_dim, d_c_prime);
    init_random_matrice(layer->W_O, d, layer->n_h * layer->v_head_dim);
    init_random_matrice(layer->W_DKV, d_c, d);

    // test sequence of inputs
    for (int step = 0; step < 3; step++)
    {
        printf("\nStep %d:\n", step);

        // create random input vector
        float *h_t = malloc(d * sizeof(float));
        init_random_vector(h_t, d);

        print_vector("Input", h_t, d);

        // Forward pass
        float *output = mla_forward(layer, h_t);

        if (output)
        {

            print_vector("Output", output, d);

            int has_nan = 0;
            for (int i = 0; i < d; i++)
            {
                if (isnan(output[i]))
                {
                    has_nan = 1;
                    break;
                }
            }

            if (has_nan)
            {
                printf("WARNING: Output contains NaN values!\n");
            }

            free(output);
        }
        else
        {
            printf("ERROR: Forward pass failed!\n");
        }

        free(h_t);
    }

    free_mla_layer(layer);

    return 0;
}

void free_matrice(float **matrice, int rows)
{
    if (matrice == NULL)
        return;
    for (int i = 0; i < rows; i++)
    {
        free(matrice[i]);
    }
    free(matrice);
}

float *dot_product(float *vec1, float *vec2, int len)
{
    float *result = (float *)malloc(len * sizeof(float));
    for (int i = 0; i < len; i++)
    {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}

float *softmax(float *vec, int len)
{
    float *result = (float *)malloc(len * sizeof(float));
    if (!result)
        return NULL;

    // find the maximum value in the vector (numerically stable softmax)
    float max_val = vec[0];
    for (int i = 1; i < len; i++)
    {
        if (vec[i] > max_val)
        {
            max_val = vec[i];
        }
    }

    // substract the maximum value and compute the exponentials
    float sum_exp = 0.0;
    for (int i = 0; i < len; i++)
    {
        result[i] = exp(vec[i] - max_val);
        sum_exp += result[i];
    }

    // normalize by the sum of exponentials
    for (int i = 0; i < len; i++)
    {
        result[i] /= sum_exp;
    }

    return result;
}

// split single vector into n_h heads
float **split_heads(float *vec, int n_h, int qk_head_dim)
{
    float **heads = allocate_matrice(n_h, qk_head_dim);
    if (!heads)
        return NULL;

    for (int i = 0; i < n_h; i++)
    {
        for (int j = 0; j < qk_head_dim; j++)
        {
            heads[i][j] = vec[i * qk_head_dim + j];
        }
    }
    return heads;
}

// heads back to a single vector
float *concat_heads(float **heads, int n_h, int qk_head_dim)
{
    float *vec = malloc(n_h * qk_head_dim * sizeof(float));
    for (int i = 0; i < n_h; i++)
    {
        for (int j = 0; j < qk_head_dim; j++)
        {
            vec[i * qk_head_dim + j] = heads[i][j];
        }
    }
    return vec;
}

float *concat_with_rope(float *vec1, float *vec2, int len1, int len2)
{
    float *result = malloc((len1 + len2) * sizeof(float));
    if (!result)
        return NULL;

    memcpy(result, vec1, len1 * sizeof(float));
    memcpy(result + len1, vec2, len2 * sizeof(float));

    return result;
}

float compute_attention_score(float *q_content, float *q_rope,
                              float *k_content, float *k_rope,
                              int head_dim, float scale)
{
    float content_score = dot_product_scalar(q_content, k_content, head_dim);
    float position_score = dot_product_scalar(q_rope, k_rope, head_dim);
    return (content_score + position_score) * scale;
}

float *mla_forward(MLA_Layer *layer, float *h_t)
{
    // allocate all intermediate tensors
    float *c_t_Q = NULL, *c_t_KV = NULL;
    float *q_t_C_nope = NULL, *q_t_C_rope = NULL;
    float *k_t_C_nope = NULL, *k_t_C_rope = NULL;
    float *v_t_C = NULL;
    float **q_t_C_heads_nope = NULL, **q_t_C_heads_rope = NULL;
    float **k_t_C_heads_nope = NULL, **k_t_C_heads_rope = NULL;
    float **v_t_C_heads = NULL;
    float **o_t_heads = NULL;
    float *o_t = NULL, *u_t = NULL;

    // use goto cleanup pattern for error handling
    // this allows us to properly free memory even if an allocation fails

    // Apply RMSNorm for query path
    float *normalized_q = RMS_norm(h_t, layer->norm_weight, layer->d);
    if (!normalized_q)
        goto cleanup;

    // Apply RMSNorm for key/value path
    float *normalized_kv = RMS_norm(h_t, layer->norm_weight, layer->d);
    if (!normalized_kv)
        goto cleanup;

    // Use normalized inputs for the respective paths
    c_t_Q = matmul(layer->W_DQ, normalized_q, layer->d_c_prime, layer->d);
    if (!c_t_Q)
        goto cleanup;

    c_t_KV = matmul(layer->W_DKV, normalized_kv, layer->d_c, layer->d);
    if (!c_t_KV)
        goto cleanup;

    //  content queries and rope queries
    q_t_C_nope = matmul(layer->W_UQ_nope, c_t_Q, layer->qk_nope_head_dim * layer->n_h, layer->d_c_prime);
    if (!q_t_C_nope)
        goto cleanup;

    q_t_C_rope = matmul(layer->W_UQ_rope, c_t_Q, layer->qk_rope_head_dim * layer->n_h, layer->d_c_prime);
    if (!q_t_C_rope)
        goto cleanup;

    // split queries into heads
    q_t_C_heads_nope = split_heads(q_t_C_nope, layer->n_h, layer->qk_nope_head_dim);
    if (!q_t_C_heads_nope)
        goto cleanup;

    q_t_C_heads_rope = split_heads(q_t_C_rope, layer->n_h, layer->qk_rope_head_dim);
    if (!q_t_C_heads_rope)
        goto cleanup;

    // apply RoPE to rope queries
    for (int i = 0; i < layer->n_h; i++)
    {
        apply_rope(q_t_C_heads_rope[i], i, layer->qk_rope_head_dim);
    }

    //  content keys and rope keys
    k_t_C_nope = matmul(layer->W_UK_nope, c_t_KV, layer->qk_nope_head_dim * layer->n_h, layer->d_c);
    if (!k_t_C_nope)
        goto cleanup;

    k_t_C_rope = matmul(layer->W_UK_rope, c_t_KV, layer->qk_rope_head_dim * layer->n_h, layer->d_c);
    if (!k_t_C_rope)
        goto cleanup;

    // Split content keys into heads
    k_t_C_heads_nope = split_heads(k_t_C_nope, layer->n_h, layer->qk_nope_head_dim);
    if (!k_t_C_heads_nope)
        goto cleanup;

    k_t_C_heads_rope = split_heads(k_t_C_rope, layer->n_h, layer->qk_rope_head_dim);
    if (!k_t_C_heads_rope)
        goto cleanup;

    //  content values
    v_t_C = matmul(layer->W_UV, c_t_KV, layer->v_head_dim * layer->n_h, layer->d_c);
    if (!v_t_C)
        goto cleanup;

    v_t_C_heads = split_heads(v_t_C, layer->n_h, layer->v_head_dim);
    if (!v_t_C_heads)
        goto cleanup;

    // allocate output heads
    o_t_heads = allocate_matrice(layer->n_h, layer->qk_nope_head_dim);
    if (!o_t_heads)
        goto cleanup;

    // process each head
    for (int i = 0; i < layer->n_h; i++)
    {
        float *attn_scores = malloc((layer->seq_len + 1) * sizeof(float));
        if (!attn_scores)
            goto cleanup;

        // calculate attention scores with all previous positions
        float max_score = -INFINITY;
        float sum_exp = 0.0f;

        // compute scores and find max
        for (int pos = 0; pos <= layer->seq_len; pos++)
        {
            float *k_content = (pos == layer->seq_len) ? k_t_C_heads_nope[i] : layer->K_cache[pos][i];

            float scale = 1.0f / sqrt(layer->qk_nope_head_dim + layer->qk_rope_head_dim);
            attn_scores[pos] = compute_attention_score(
                q_t_C_heads_nope[i], q_t_C_heads_rope[i],
                k_content, k_t_C_heads_rope[i],
                layer->qk_nope_head_dim, scale);

            if (attn_scores[pos] > max_score)
            {
                max_score = attn_scores[pos];
            }
        }

        // compute exponentials and sum
        for (int pos = 0; pos <= layer->seq_len; pos++)
        {
            attn_scores[pos] = exp(attn_scores[pos] - max_score);
            sum_exp += attn_scores[pos];
        }

        for (int j = 0; j < layer->qk_nope_head_dim; j++)
        {
            o_t_heads[i][j] = 0;
        }

        for (int pos = 0; pos <= layer->seq_len; pos++)
        {
            float weight = attn_scores[pos] / sum_exp;
            float *v_head = (pos == layer->seq_len) ? v_t_C_heads[i] : layer->V_cache[pos][i];

            for (int j = 0; j < layer->qk_nope_head_dim; j++)
            {
                o_t_heads[i][j] += weight * v_head[j];
            }
        }

        if (layer->seq_len < layer->max_seq_len)
        {
            memcpy(layer->K_cache[layer->seq_len][i], k_t_C_heads_nope[i],
                   layer->qk_nope_head_dim * sizeof(float));
            memcpy(layer->V_cache[layer->seq_len][i], v_t_C_heads[i],
                   layer->v_head_dim * sizeof(float));
        }

        free(attn_scores);
    }

    if (layer->seq_len < layer->max_seq_len)
    {
        layer->seq_len++;
    }

    o_t = concat_heads(o_t_heads, layer->n_h, layer->qk_nope_head_dim);
    if (!o_t)
        goto cleanup;

    u_t = matmul(layer->W_O, o_t, layer->d, layer->n_h * layer->v_head_dim);

cleanup:
    free(c_t_Q);
    free(c_t_KV);
    free(q_t_C_nope);
    free(q_t_C_rope);
    free(k_t_C_nope);
    free(k_t_C_rope);
    free(v_t_C);

    if (q_t_C_heads_nope)
        free_matrice(q_t_C_heads_nope, layer->n_h);
    if (q_t_C_heads_rope)
        free_matrice(q_t_C_heads_rope, layer->n_h);
    if (k_t_C_heads_nope)
        free_matrice(k_t_C_heads_nope, layer->n_h);
    if (k_t_C_heads_rope)
        free_matrice(k_t_C_heads_rope, layer->n_h);
    if (v_t_C_heads)
        free_matrice(v_t_C_heads, layer->n_h);
    if (o_t_heads)
        free_matrice(o_t_heads, layer->n_h);

    free(o_t);

    // Add new normalizations to cleanup
    free(normalized_q);
    free(normalized_kv);

    return u_t;
}