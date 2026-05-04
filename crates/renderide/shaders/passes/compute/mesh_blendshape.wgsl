// Sparse weighted blendshape deltas. The encoder dispatches one channel at a
// time so sparse position, normal, and tangent storage can stay channel-local.

const CHANNEL_POSITION: u32 = 0u;
const CHANNEL_NORMAL: u32 = 1u;
const CHANNEL_TANGENT: u32 = 2u;
const POSITION_ENTRY_WORDS: u32 = 4u;
const PACKED_VECTOR_ENTRY_WORDS: u32 = 3u;
const PACKED_VECTOR_DELTA_RANGE: f32 = 2.0;

struct Params {
    vertex_count: u32,
    sparse_base_word: u32,
    sparse_count: u32,
    base_dst_e: u32,
    channel: u32,
    effective_weight: f32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sparse_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_vec: array<vec4<f32>>;

fn unpack_snorm16_delta(bits: u32) -> f32 {
    let raw = bits & 0xffffu;
    let signed = select(i32(raw), i32(raw) - 65536, (raw & 0x8000u) != 0u);
    return max(f32(signed) / 32767.0, -1.0) * PACKED_VECTOR_DELTA_RANGE;
}

fn load_position_delta(entry_word: u32) -> vec3<f32> {
    return vec3<f32>(
        bitcast<f32>(sparse_words[entry_word + 1u]),
        bitcast<f32>(sparse_words[entry_word + 2u]),
        bitcast<f32>(sparse_words[entry_word + 3u]),
    );
}

fn load_packed_vector_delta(entry_word: u32) -> vec3<f32> {
    let xy = sparse_words[entry_word + 1u];
    let z = sparse_words[entry_word + 2u];
    return vec3<f32>(
        unpack_snorm16_delta(xy),
        unpack_snorm16_delta(xy >> 16u),
        unpack_snorm16_delta(z),
    );
}

fn entry_word_stride() -> u32 {
    if (params.channel == CHANNEL_POSITION) {
        return POSITION_ENTRY_WORDS;
    }
    return PACKED_VECTOR_ENTRY_WORDS;
}

fn load_delta(entry_word: u32) -> vec3<f32> {
    if (params.channel == CHANNEL_POSITION) {
        return load_position_delta(entry_word);
    }
    return load_packed_vector_delta(entry_word);
}

@compute @workgroup_size(64)
fn blendshape_scatter_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.sparse_count || params.effective_weight == 0.0) {
        return;
    }

    let entry_word = params.sparse_base_word + i * entry_word_stride();
    let vi = sparse_words[entry_word];
    if (vi >= params.vertex_count) {
        return;
    }

    let oi = params.base_dst_e + vi;
    let current = out_vec[oi];
    out_vec[oi] = vec4<f32>(
        current.xyz + params.effective_weight * load_delta(entry_word),
        current.w,
    );
}
