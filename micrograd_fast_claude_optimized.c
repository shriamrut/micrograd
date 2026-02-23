#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

RNG rng = {42};

// ------------------------------------------------------------
// Arena Allocator
// Pre-allocate a big flat buffer. Each "alloc" just bumps a pointer.
// At end of each training step, reset offset to 0 — free everything instantly.
// No malloc overhead, no fragmentation, great cache locality.

#define ARENA_SIZE (1024 * 1024 * 64) // 64MB — plenty for our graph

typedef struct Arena {
    char *buf;
    size_t offset;
    size_t capacity;
} Arena;

Arena* arena_new(size_t capacity) {
    Arena *a = malloc(sizeof(Arena));
    a->buf = malloc(capacity);
    a->offset = 0;
    a->capacity = capacity;
    return a;
}

void* arena_alloc(Arena *a, size_t size) {
    // align to 8 bytes
    size = (size + 7) & ~7;
    if (a->offset + size > a->capacity) {
        fprintf(stderr, "Arena out of memory! Increase ARENA_SIZE.\n");
        exit(1);
    }
    void *ptr = a->buf + a->offset;
    a->offset += size;
    return ptr;
}

void arena_reset(Arena *a) {
    a->offset = 0;
}

// Global arena — used for all per-step allocations (Values, children arrays, topo array)
Arena *step_arena;

// ------------------------------------------------------------
// Value

typedef struct Value {
    double data;
    double grad;
    double m;       // adamw moment 1
    double v;       // adamw moment 2
    struct Value **children;  // NULL if leaf
    int n_children;
    void (*_backward)(struct Value*);
    bool visited;   // for topo sort — replaces O(n^2) is_present scan
} Value;

void print_value(Value *a) {
    printf("Value(data=%f, grad=%f)\n", a->data, a->grad);
}

// Allocate Value from arena (fast, no malloc)
Value* new_value(double data, Value **children, int n_children) {
    Value *val = arena_alloc(step_arena, sizeof(Value));
    val->data = data;
    val->grad = 0.0;
    val->m = 0.0;
    val->v = 0.0;
    val->children = children;
    val->n_children = n_children;
    val->_backward = NULL;
    val->visited = false;
    return val;
}

// Allocate a children array from arena
Value** new_children(int n) {
    return arena_alloc(step_arena, n * sizeof(Value*));
}

// ------------------------------------------------------------
// Backward functions

void add_backward(Value *out) {
    out->children[0]->grad += out->grad;
    out->children[1]->grad += out->grad;
}

void multiply_backward(Value *out) {
    out->children[0]->grad += out->children[1]->data * out->grad;
    out->children[1]->grad += out->children[0]->data * out->grad;
}

void power_backward(Value *out) {
    // children[0]=base, children[1]=exponent (constant)
    Value *base = out->children[0];
    double n = out->children[1]->data;
    base->grad += n * pow(base->data, n - 1) * out->grad;
}

void log_backward(Value *out) {
    out->children[0]->grad += (1.0 / out->children[0]->data) * out->grad;
}

void tanh_backward(Value *out) {
    out->children[0]->grad += (1.0 - out->data * out->data) * out->grad;
}

void exp_backward(Value *out) {
    out->children[0]->grad += exp(out->children[0]->data) * out->grad;
}

// ------------------------------------------------------------
// Forward ops — all allocate from arena

Value* multiply(Value *a, Value *b) {
    Value **ch = new_children(2);
    ch[0] = a; ch[1] = b;
    Value *out = new_value(a->data * b->data, ch, 2);
    out->_backward = multiply_backward;
    return out;
}

Value* add(Value *a, Value *b) {
    Value **ch = new_children(2);
    ch[0] = a; ch[1] = b;
    Value *out = new_value(a->data + b->data, ch, 2);
    out->_backward = add_backward;
    return out;
}

Value* power(Value *base, double exp_val) {
    // exponent is a constant — store as a leaf child so backward can read it
    Value **ch = new_children(2);
    ch[0] = base;
    ch[1] = new_value(exp_val, NULL, 0); // constant, no backward needed
    Value *out = new_value(pow(base->data, exp_val), ch, 2);
    out->_backward = power_backward;
    return out;
}

Value* negative(Value *a) {
    Value *neg_one = new_value(-1.0, NULL, 0);
    return multiply(neg_one, a);
}

Value* subtract(Value *a, Value *b) {
    return add(a, negative(b));
}

Value* log_value(Value *a) {
    Value **ch = new_children(1);
    ch[0] = a;
    Value *out = new_value(log(a->data), ch, 1);
    out->_backward = log_backward;
    return out;
}

Value* exp_value(Value *a) {
    Value **ch = new_children(1);
    ch[0] = a;
    Value *out = new_value(exp(a->data), ch, 1);
    out->_backward = exp_backward;
    return out;
}

Value* true_div(Value *a, Value *b) {
    return multiply(a, power(b, -1.0));
}

Value* tanh_act(Value *a) {
    Value **ch = new_children(1);
    ch[0] = a;
    Value *out = new_value(tanh(a->data), ch, 1);
    out->_backward = tanh_backward;
    return out;
}

// ------------------------------------------------------------
// Topological sort — flat array, O(n) with visited flag on Value

// We'll use a dynamic array that also lives in the arena
typedef struct TopoArray {
    Value **nodes;
    int size;
    int capacity;
} TopoArray;

TopoArray* topo_new(int initial_capacity) {
    TopoArray *t = arena_alloc(step_arena, sizeof(TopoArray));
    t->nodes = arena_alloc(step_arena, initial_capacity * sizeof(Value*));
    t->size = 0;
    t->capacity = initial_capacity;
    return t;
}

void topo_push(TopoArray *t, Value *v) {
    if (t->size >= t->capacity) {
        // This shouldn't happen if initial_capacity is large enough
        fprintf(stderr, "TopoArray overflow! Increase initial capacity.\n");
        exit(1);
    }
    t->nodes[t->size++] = v;
}

void build_topo(Value *v, TopoArray *topo) {
    if (v == NULL || v->visited) return;
    v->visited = true;
    for (int i = 0; i < v->n_children; i++) {
        build_topo(v->children[i], topo);
    }
    topo_push(topo, v);
}

void backward(Value *out) {
    // 8192 nodes is plenty for our graph (80 points * ~50 ops + overhead)
    TopoArray *topo = topo_new(8192 * 16);
    build_topo(out, topo);

    out->grad = 1.0;
    // traverse in reverse (topo is sorted leaves-first, we want root-first)
    for (int i = topo->size - 1; i >= 0; i--) {
        Value *v = topo->nodes[i];
        if (v->_backward != NULL) {
            v->_backward(v);
        }
    }
}

// ------------------------------------------------------------
// MLP — weights/biases live OUTSIDE the step arena (they persist across steps)
// Only the computation graph (forward pass Values) lives in the step arena.

typedef struct Neuron {
    double *w;      // raw doubles — not Values
    double b;
    double *w_m, *w_v;  // adamw moments for weights
    double b_m, b_v;    // adamw moments for bias
    int nin;
    bool nonlin;
} Neuron;

typedef struct Layer {
    Neuron *neurons;
    int nout;
} Layer;

typedef struct MLP {
    Layer *layers;
    int layer_count;
    int *sizes; // sizes[0]=nin, sizes[1..n]=nouts
} MLP;

// Parameter struct — flat array of pointers into neuron weight/bias storage
// Built once before training, reused every step
typedef struct ParamRef {
    double *data;
    double *m;
    double *v;
    double *grad; // points into a per-step grad accumulation array
} ParamRef;

// We'll store gradients in a flat array that gets zeroed each step
typedef struct Params {
    ParamRef *refs;
    double *grads;  // flat array, same count as refs
    int count;
} Params;

Neuron new_neuron(int nin, bool nonlin) {
    Neuron n;
    n.nin = nin;
    n.nonlin = nonlin;
    n.w   = malloc(nin * sizeof(double));
    n.w_m = calloc(nin, sizeof(double));
    n.w_v = calloc(nin, sizeof(double));
    n.b   = 0.0;
    n.b_m = 0.0;
    n.b_v = 0.0;
    for (int i = 0; i < nin; i++) {
        n.w[i] = rng_uniform(&rng, -1.0, 1.0) / sqrt(nin);
    }
    return n;
}

Layer new_layer(int nin, int nout, bool nonlin) {
    Layer l;
    l.nout = nout;
    l.neurons = malloc(nout * sizeof(Neuron));
    for (int i = 0; i < nout; i++) {
        l.neurons[i] = new_neuron(nin, nonlin);
    }
    return l;
}

MLP* new_mlp(int nin, int *nouts, int n_layers) {
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layer_count = n_layers;
    mlp->layers = malloc(n_layers * sizeof(Layer));
    mlp->sizes = malloc((n_layers + 1) * sizeof(int));
    mlp->sizes[0] = nin;
    for (int i = 0; i < n_layers; i++) {
        mlp->sizes[i+1] = nouts[i];
    }
    for (int i = 0; i < n_layers; i++) {
        mlp->layers[i] = new_layer(mlp->sizes[i], mlp->sizes[i+1], i != n_layers - 1);
    }
    return mlp;
}

// Build the flat parameter reference list once
Params* build_params(MLP *mlp) {
    // count total params
    int count = 0;
    for (int i = 0; i < mlp->layer_count; i++) {
        Layer *l = &mlp->layers[i];
        for (int j = 0; j < l->nout; j++) {
            count += l->neurons[j].nin + 1; // weights + bias
        }
    }

    Params *p = malloc(sizeof(Params));
    p->count = count;
    p->refs  = malloc(count * sizeof(ParamRef));
    p->grads = calloc(count, sizeof(double));

    int idx = 0;
    for (int i = 0; i < mlp->layer_count; i++) {
        Layer *l = &mlp->layers[i];
        for (int j = 0; j < l->nout; j++) {
            Neuron *n = &l->neurons[j];
            for (int k = 0; k < n->nin; k++) {
                p->refs[idx].data = &n->w[k];
                p->refs[idx].m    = &n->w_m[k];
                p->refs[idx].v    = &n->w_v[k];
                p->refs[idx].grad = &p->grads[idx];
                idx++;
            }
            p->refs[idx].data = &n->b;
            p->refs[idx].m    = &n->b_m;
            p->refs[idx].v    = &n->b_v;
            p->refs[idx].grad = &p->grads[idx];
            idx++;
        }
    }
    return p;
}

void zero_grads(Params *p) {
    memset(p->grads, 0, p->count * sizeof(double));
}

// ------------------------------------------------------------
// Forward pass — creates Value nodes in the step arena
// Weights/biases are wrapped in temporary leaf Values each step

Value** forward_neuron(Neuron *n, Value **in, Params *params, int param_offset) {
    // wrap weight doubles as leaf Values pointing grad back to params->grads
    // We'll do this more simply: create leaf Values, then after backward,
    // harvest grads from them. We track them via the param refs.
    // Actually simpler: build leaf Values for weights/bias, do forward, backward gives grads on them.
    // Then copy grads to param->grads array.
    // But that requires finding each leaf Value... 
    // 
    // Cleaner approach: leaf Values have their grad written by backward. 
    // We store pointers to those leaf Values so we can read their grads after backward.
    // Store them in a flat array per step — but that complicates things.
    //
    // SIMPLEST: just use Value leaves normally, harvest grad after backward by index.
    // We'll return an array containing the output Value + store weight/bias Value pointers
    // in a per-step flat array indexed by param_offset.
    // See forward_model below for how this is orchestrated.
    return NULL; // placeholder, see forward_model
}

// We'll use a simpler and cleaner design:
// Store Value* leaves for all params in a flat array each step.
// After backward, copy their .grad into params->grads.

Value** forward_model_full(MLP *mlp, Value **input, Value **param_leaves) {
    // param_leaves: pre-allocated flat array of Value* for all params this step
    // We fill it in layer/neuron/weight order (same as build_params)
    int pidx = 0;

    Value **current = input;
    for (int li = 0; li < mlp->layer_count; li++) {
        Layer *l = &mlp->layers[li];
        Value **next = arena_alloc(step_arena, l->nout * sizeof(Value*));
        for (int ni = 0; ni < l->nout; ni++) {
            Neuron *n = &l->neurons[ni];
            // create leaf Values for weights and bias
            Value *act = NULL;
            for (int wi = 0; wi < n->nin; wi++) {
                Value *w_val = new_value(n->w[wi], NULL, 0);
                param_leaves[pidx++] = w_val;
                Value *term = multiply(w_val, current[wi]);
                act = (act == NULL) ? term : add(act, term);
            }
            Value *b_val = new_value(n->b, NULL, 0);
            param_leaves[pidx++] = b_val;
            act = add(act, b_val);
            next[ni] = n->nonlin ? tanh_act(act) : act;
        }
        current = next;
    }
    return current;
}

// ------------------------------------------------------------
// Cross entropy loss (fixed: sum starts with elements[0] directly)

Value* cross_entropy(Value **logits, int target, int size) {
    // subtract max for numerical stability
    double max_val = logits[0]->data;
    for (int i = 1; i < size; i++) {
        if (logits[i]->data > max_val) max_val = logits[i]->data;
    }
    Value *max_v = new_value(max_val, NULL, 0); // constant, detached

    Value **shifted = arena_alloc(step_arena, size * sizeof(Value*));
    for (int i = 0; i < size; i++) {
        shifted[i] = subtract(logits[i], max_v);
    }

    Value **ex = arena_alloc(step_arena, size * sizeof(Value*));
    for (int i = 0; i < size; i++) {
        ex[i] = exp_value(shifted[i]);
    }

    // FIX: start sum with ex[0] directly (not a copy), so it stays in the graph
    Value *denom = ex[0];
    for (int i = 1; i < size; i++) {
        denom = add(denom, ex[i]);
    }

    Value **probs = arena_alloc(step_arena, size * sizeof(Value*));
    for (int i = 0; i < size; i++) {
        probs[i] = true_div(ex[i], denom);
    }

    Value *logp = log_value(probs[target]);
    return negative(logp);
}

// ------------------------------------------------------------
// AdamW update

void adamw_update(Params *p, double lr, double beta1, double beta2,
                  double weight_decay, int step) {
    for (int i = 0; i < p->count; i++) {
        double g = p->grads[i];
        *p->refs[i].m = beta1 * (*p->refs[i].m) + (1 - beta1) * g;
        *p->refs[i].v = beta2 * (*p->refs[i].v) + (1 - beta2) * g * g;
        double m_hat = *p->refs[i].m / (1 - pow(beta1, step + 1));
        double v_hat = *p->refs[i].v / (1 - pow(beta2, step + 1));
        *p->refs[i].data -= lr * (m_hat / (sqrt(v_hat) + 1e-8) + weight_decay * (*p->refs[i].data));
    }
}

// ------------------------------------------------------------
// Eval split

double eval_split(MLP *mlp, DataPoint **split, int size, Params *params) {
    int n_params = params->count;
    Value **param_leaves = arena_alloc(step_arena, n_params * sizeof(Value*));

    double total_loss = 0.0;
    for (int i = 0; i < size; i++) {
        // reset arena for each point during eval (we don't need backward)
        Value **input = arena_alloc(step_arena, 2 * sizeof(Value*));
        input[0] = new_value(split[i]->x, NULL, 0);
        input[1] = new_value(split[i]->y, NULL, 0);
        Value **logits = forward_model_full(mlp, input, param_leaves);
        Value *loss = cross_entropy(logits, split[i]->label, 3);
        total_loss += loss->data;
    }
    return total_loss / size;
}

// ------------------------------------------------------------
// Main

int main() {
    // Init arena
    step_arena = arena_new(ARENA_SIZE);

    // Generate dataset
    DataSet *dataset = generate(100, &rng);

    int nouts[] = {16, 3};
    MLP *mlp = new_mlp(2, nouts, 2);

    // Build param list once — reused every step
    Params *params = build_params(mlp);

    double lr = 1e-1, beta1 = 0.9, beta2 = 0.95, weight_decay = 1e-4;
    int iterations = 100;

    for (int step = 0; step < iterations; step++) {

        // Reset arena — frees entire previous step's graph instantly
        arena_reset(step_arena);

        if (step % 10 == 0) {
            double val_loss = eval_split(mlp, dataset->val, dataset->val_size, params);
            printf("step %d, val loss %.6f\n", step, val_loss);
            arena_reset(step_arena); // reset after eval too
        }

        // Each forward pass creates fresh leaf Values for params.
        // We accumulate gradients across all points into params->grads manually.
        // Strategy: build one big graph for all points (sum of losses), 
        // then backward once. param_leaves is overwritten each point,
        // so we must collect all leaf Value* pointers across all points.

        // Allocate a 2D array: param_leaf_per_point[point][param_idx]
        Value ***all_param_leaves = arena_alloc(step_arena,
            dataset->tr_size * sizeof(Value**));
        for (int i = 0; i < dataset->tr_size; i++) {
            all_param_leaves[i] = arena_alloc(step_arena,
                params->count * sizeof(Value*));
        }

        // Forward pass over all training points — one connected graph
        Value *loss = new_value(0.0, NULL, 0);
        for (int i = 0; i < dataset->tr_size; i++) {
            Value **input = arena_alloc(step_arena, 2 * sizeof(Value*));
            input[0] = new_value(dataset->tr[i]->x, NULL, 0);
            input[1] = new_value(dataset->tr[i]->y, NULL, 0);
            Value **logits = forward_model_full(mlp, input, all_param_leaves[i]);
            Value *pt_loss = cross_entropy(logits, dataset->tr[i]->label, 3);
            loss = add(loss, pt_loss);
        }
        loss = true_div(loss, new_value((double)dataset->tr_size, NULL, 0));

        // Backward — grads flow to all leaf Values across all points
        backward(loss);

        // Harvest gradients: sum across all points for each param
        zero_grads(params);
        for (int i = 0; i < dataset->tr_size; i++) {
            for (int j = 0; j < params->count; j++) {
                params->grads[j] += all_param_leaves[i][j]->grad;
            }
        }

        // AdamW update
        adamw_update(params, lr, beta1, beta2, weight_decay, step);

        printf("step %d, train loss %.6f\n", step, loss->data);
    }

    return 0;
}
