#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

// Forward declarations
typedef struct MLA_Layer
{
    int d;
    int n_h;
    int d_h;
    int d_R_h;        // RoPE dimension per head
    int d_c;          // KV compression dimension
    int d_c_prime;    // query compression dimension
    float **W_DQ;     // down-projection matrix for queries
    float **W_UQ;     // up-projection matrix for queries
    float **W_UK;     // up-projection matrix for keys
    float **W_UV;     // up-projection matrix for values
    float **W_O;      // output projection matrix
    float **W_DKV;    // down-projection matrix for keys and values
    float **W_KR;     // rope rotation matrix for keys
    float **W_QR;     // RoPE matrix for queries (d_R_h*n_h × d_c_prime)
    float ***K_cache; // Cache for keys [seq_len][n_h][d_h]
    float ***V_cache; // Cache for values [seq_len][n_h][d_h]
    int seq_len;      // Current sequence length
    int max_seq_len;  // Maximum sequence length
} MLA_Layer;
void init_random_matrix(float **matrix, int rows, int cols);
void init_random_vector(float *vec, int size);
void print_vector(const char *name, float *vec, int size);
MLA_Layer *init_mla_layer(int d, int n_h, int d_h, int d_c, int d_c_prime, int d_R_h, int max_seq_len);
void free_mla_layer(MLA_Layer *layer);
float *mla_forward(MLA_Layer *layer, float *h_t);
float *matmul(float **A, float *x, int m, int n);
void apply_rope(float *vec, int position, int d_h);
float **allocate_matrix(int rows, int cols);
float dot_product_scalar(float *vec1, float *vec2, int len);
void free_matrix(float **matrix, int rows);
float *concat_with_rope(float *vec1, float *vec2, int len1, int len2);

void init_random_matrix(float **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Values between -1 and 1
        }
    }
}

// Helper function to initialize a vector with random values
void init_random_vector(float *vec, int size)
{
    for (int i = 0; i < size; i++)
    {
        vec[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // Values between -1 and 1
    }
}

// Helper function to print a vector
void print_vector(const char *name, float *vec, int size)
{
    printf("%s: [", name);
    for (int i = 0; i < size && i < 5; i++)
    { // Print first 5 elements
        printf("%.4f", vec[i]);
        if (i < size - 1 && i < 4)
            printf(", ");
    }
    if (size > 5)
        printf(", ...");
    printf("]\n");
}

MLA_Layer *init_mla_layer(int d, int n_h, int d_h, int d_c, int d_c_prime, int d_R_h, int max_seq_len)
{
    MLA_Layer *layer = (MLA_Layer *)malloc(sizeof(MLA_Layer));
    layer->d = d;
    layer->n_h = n_h;
    layer->d_h = d_h;
    layer->d_c = d_c;
    layer->d_c_prime = d_c_prime;
    layer->d_R_h = d_R_h;
    // projection matrices
    layer->W_DQ = allocate_matrix(d_c_prime, d);
    layer->W_UQ = allocate_matrix(d_h * n_h, d_c_prime);
    layer->W_UK = allocate_matrix(d_h * n_h, d_c);
    layer->W_UV = allocate_matrix(d_h * n_h, d_c);
    layer->W_O = allocate_matrix(d, d_h * n_h);
    layer->W_DKV = allocate_matrix(d_c, d);
    layer->W_KR = allocate_matrix(d_R_h, d);
    layer->W_QR = allocate_matrix(d_R_h * n_h, d_c_prime);

    // Initialize caches
    layer->max_seq_len = max_seq_len;
    layer->seq_len = 0;
    layer->K_cache = malloc(max_seq_len * sizeof(float **));
    layer->V_cache = malloc(max_seq_len * sizeof(float **));
    for (int i = 0; i < max_seq_len; i++)
    {
        layer->K_cache[i] = allocate_matrix(n_h, d_h);
        layer->V_cache[i] = allocate_matrix(n_h, d_h);
    }

    return layer;
}

void free_mla_layer(MLA_Layer *layer)
{
    if (layer == NULL)
        return;

    // Free all projection matrices
    free_matrix(layer->W_DQ, layer->d_c_prime);
    free_matrix(layer->W_UQ, layer->d_h * layer->n_h);
    free_matrix(layer->W_UK, layer->d_h * layer->n_h);
    free_matrix(layer->W_UV, layer->d_h * layer->n_h);
    free_matrix(layer->W_O, layer->d);
    free_matrix(layer->W_DKV, layer->d_c);
    free_matrix(layer->W_KR, layer->d_R_h);
    free_matrix(layer->W_QR, layer->d_R_h * layer->n_h);

    // Free the layer struct itself
    free(layer);
}

// Modified matmul with error checking
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

float **allocate_matrix(int rows, int cols)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
    }
    return matrix;
}

void apply_rope(float *vec, int position, int d_h)
{
    for (int i = 0; i < d_h; i += 2)
    {
        float theta = position / pow(10000.0, (float)i / d_h);
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);

        float x0 = vec[i];
        float x1 = vec[i + 1];

        vec[i] = x0 * cos_theta - x1 * sin_theta;
        vec[i + 1] = x0 * sin_theta + x1 * cos_theta;
    }
}

// Add helper function for scalar dot product
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
    int d_h = 64;
    int d_c = 128;
    int d_c_prime = 128;
    int d_R_h = 32;
    int max_seq_len = 4;

    // Initialize MLA layer
    MLA_Layer *layer = init_mla_layer(d, n_h, d_h, d_c, d_c_prime, d_R_h, max_seq_len);
    if (!layer)
    {
        printf("Failed to initialize MLA layer\n");
        return 1;
    }

    // Initialize weights with random values
    init_random_matrix(layer->W_DQ, d_c_prime, d);
    init_random_matrix(layer->W_UQ, d_h * n_h, d_c_prime);
    init_random_matrix(layer->W_UK, d_h * n_h, d_c);
    init_random_matrix(layer->W_UV, d_h * n_h, d_c);
    init_random_matrix(layer->W_O, d, d_h * n_h);
    init_random_matrix(layer->W_DKV, d_c, d);
    init_random_matrix(layer->W_KR, d_R_h, d);
    init_random_matrix(layer->W_QR, d_R_h * n_h, d_c_prime);

    // Test sequence of inputs
    for (int step = 0; step < 3; step++)
    {
        printf("\nStep %d:\n", step);

        // Create random input vector
        float *h_t = malloc(d * sizeof(float));
        init_random_vector(h_t, d);

        // Print input
        print_vector("Input", h_t, d);

        // Forward pass
        float *output = mla_forward(layer, h_t);

        if (output)
        {
            // Print output
            print_vector("Output", output, d);

            // Check for NaN values
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

            // Free output
            free(output);
        }
        else
        {
            printf("ERROR: Forward pass failed!\n");
        }

        // Free input
        free(h_t);
    }

    // Free MLA layer
    free_mla_layer(layer);

    return 0;
}

void free_matrix(float **matrix, int rows)
{
    if (matrix == NULL)
        return;
    for (int i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
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

    // Find the maximum value in the vector
    float max_val = vec[0];
    for (int i = 1; i < len; i++)
    {
        if (vec[i] > max_val)
        {
            max_val = vec[i];
        }
    }

    // Subtract the maximum value and compute the exponentials
    float sum_exp = 0.0;
    for (int i = 0; i < len; i++)
    {
        result[i] = exp(vec[i] - max_val);
        sum_exp += result[i];
    }

    // Normalize by the sum of exponentials
    for (int i = 0; i < len; i++)
    {
        result[i] /= sum_exp;
    }

    return result;
}

// Modified split_heads with error checking
float **split_heads(float *vec, int n_h, int d_h)
{
    float **heads = allocate_matrix(n_h, d_h);
    if (!heads)
        return NULL;

    for (int i = 0; i < n_h; i++)
    {
        for (int j = 0; j < d_h; j++)
        {
            heads[i][j] = vec[i * d_h + j];
        }
    }
    return heads;
}

// heads back to a single vector
float *concat_heads(float **heads, int n_h, int d_h)
{
    float *vec = malloc(n_h * d_h * sizeof(float));
    for (int i = 0; i < n_h; i++)
    {
        for (int j = 0; j < d_h; j++)
        {
            vec[i * d_h + j] = heads[i][j];
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

float *mla_forward(MLA_Layer *layer, float *h_t)
{
    // Allocate all intermediate tensors
    float *c_t_Q = NULL, *c_t_KV = NULL;
    float *q_t_C = NULL, *q_t_R = NULL;
    float *k_t_C = NULL, *k_t_R = NULL;
    float *v_t_C = NULL;
    float **q_t_C_heads = NULL, **q_t_R_heads = NULL;
    float **k_t_C_heads = NULL, **v_t_C_heads = NULL;
    float **o_t_heads = NULL;
    float *o_t = NULL, *u_t = NULL;

    // Use goto cleanup pattern for error handling
    // This allows us to properly free memory even if an allocation fails

    // Compute compressed latent vectors
    c_t_Q = matmul(layer->W_DQ, h_t, layer->d_c_prime, layer->d);
    if (!c_t_Q)
        goto cleanup;

    c_t_KV = matmul(layer->W_DKV, h_t, layer->d_c, layer->d);
    if (!c_t_KV)
        goto cleanup;

    // Compute content queries and rope queries
    q_t_C = matmul(layer->W_UQ, c_t_Q, layer->d_h * layer->n_h, layer->d_c_prime);
    if (!q_t_C)
        goto cleanup;

    q_t_R = matmul(layer->W_QR, c_t_Q, layer->d_R_h * layer->n_h, layer->d_c_prime);
    if (!q_t_R)
        goto cleanup;

    // Split queries into heads
    q_t_C_heads = split_heads(q_t_C, layer->n_h, layer->d_h);
    if (!q_t_C_heads)
        goto cleanup;

    q_t_R_heads = split_heads(q_t_R, layer->n_h, layer->d_h);
    if (!q_t_R_heads)
        goto cleanup;

    // Apply RoPE to rope queries
    for (int i = 0; i < layer->n_h; i++)
    {
        apply_rope(q_t_R_heads[i], i, layer->d_R_h);
    }

    // Compute content keys and rope keys
    k_t_C = matmul(layer->W_UK, c_t_KV, layer->d_h * layer->n_h, layer->d_c);
    if (!k_t_C)
        goto cleanup;

    k_t_R = matmul(layer->W_KR, h_t, layer->d_R_h, layer->d);
    if (!k_t_R)
        goto cleanup;

    // Split content keys into heads
    k_t_C_heads = split_heads(k_t_C, layer->n_h, layer->d_h);
    if (!k_t_C_heads)
        goto cleanup;

    // Apply RoPE to rotational keys
    apply_rope(k_t_R, 0, layer->d_R_h);

    // Compute content values
    v_t_C = matmul(layer->W_UV, c_t_KV, layer->d_h * layer->n_h, layer->d_c);
    if (!v_t_C)
        goto cleanup;

    v_t_C_heads = split_heads(v_t_C, layer->n_h, layer->d_h);
    if (!v_t_C_heads)
        goto cleanup;

    // Allocate output heads
    o_t_heads = allocate_matrix(layer->n_h, layer->d_h);
    if (!o_t_heads)
        goto cleanup;

    float scaling_factor = 1.0 / sqrt(layer->d_h + layer->d_R_h);

    // Process each head
    for (int i = 0; i < layer->n_h; i++)
    {
        float *attn_scores = malloc((layer->seq_len + 1) * sizeof(float));
        if (!attn_scores)
            goto cleanup;

        // Current position query
        float *q_head = concat_with_rope(q_t_C_heads[i], q_t_R_heads[i],
                                         layer->d_h, layer->d_R_h);
        if (!q_head)
        {
            free(attn_scores);
            goto cleanup;
        }

        // Calculate attention scores with all previous positions
        float sum_exp = 0.0;
        float max_score = -INFINITY;

        // First, compute all scores and find max
        for (int pos = 0; pos <= layer->seq_len; pos++)
        {
            float *k_head;
            if (pos == layer->seq_len)
            {
                // Current position
                k_head = concat_with_rope(k_t_C_heads[i], k_t_R,
                                          layer->d_h, layer->d_R_h);
            }
            else
            {
                // Previous position from cache
                k_head = layer->K_cache[pos][i];
            }

            attn_scores[pos] = dot_product_scalar(q_head, k_head,
                                                  layer->d_h + layer->d_R_h) /
                               sqrt(layer->d_h + layer->d_R_h);

            if (attn_scores[pos] > max_score)
            {
                max_score = attn_scores[pos];
            }
        }

        // Apply softmax and compute weighted sum
        for (int pos = 0; pos <= layer->seq_len; pos++)
        {
            attn_scores[pos] = exp(attn_scores[pos] - max_score);
            sum_exp += attn_scores[pos];
        }

        // Initialize output head values to 0
        for (int j = 0; j < layer->d_h; j++)
        {
            o_t_heads[i][j] = 0;
        }

        // Compute weighted sum of values
        for (int pos = 0; pos <= layer->seq_len; pos++)
        {
            float weight = attn_scores[pos] / sum_exp;
            float *v_head = (pos == layer->seq_len) ? v_t_C_heads[i] : layer->V_cache[pos][i];

            for (int j = 0; j < layer->d_h; j++)
            {
                o_t_heads[i][j] += weight * v_head[j];
            }
        }

        // Cache current key and value
        if (layer->seq_len < layer->max_seq_len)
        {
            memcpy(layer->K_cache[layer->seq_len][i], k_t_C_heads[i],
                   layer->d_h * sizeof(float));
            memcpy(layer->V_cache[layer->seq_len][i], v_t_C_heads[i],
                   layer->d_h * sizeof(float));
        }

        free(q_head);
        free(attn_scores);
    }

    // Increment sequence length
    if (layer->seq_len < layer->max_seq_len)
    {
        layer->seq_len++;
    }

    // Concatenate all heads
    o_t = concat_heads(o_t_heads, layer->n_h, layer->d_h);
    if (!o_t)
        goto cleanup;

    // Final output projection
    u_t = matmul(layer->W_O, o_t, layer->d, layer->d_h * layer->n_h);

cleanup:
    // Free all intermediate tensors
    free(c_t_Q);
    free(c_t_KV);
    free(q_t_C);
    free(q_t_R);
    free(k_t_C);
    free(k_t_R);
    free(v_t_C);

    if (q_t_C_heads)
        free_matrix(q_t_C_heads, layer->n_h);
    if (q_t_R_heads)
        free_matrix(q_t_R_heads, layer->n_h);
    if (k_t_C_heads)
        free_matrix(k_t_C_heads, layer->n_h);
    if (v_t_C_heads)
        free_matrix(v_t_C_heads, layer->n_h);
    if (o_t_heads)
        free_matrix(o_t_heads, layer->n_h);

    free(o_t); // u_t will be freed by the caller

    return u_t; // Could be NULL if any allocation failed
}
