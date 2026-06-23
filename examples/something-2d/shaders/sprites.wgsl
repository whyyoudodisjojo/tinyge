struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

struct SpriteData{
    pos: vec2<f32>,
    scale: vec2<f32>,
    uv_offset: vec2<f32>,
    uv_scale: vec2<f32>
};

@group(0) @binding(0) var<uniform> time: f32;
@group(0) @binding(1) var<storage, read> sprite_data: array<SpriteData>;

@group(1) @binding(0) var sprite_sampler: sampler;
@group(1) @binding(1) var sprite_texture: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32, @builtin(instance_index) i_idx: u32) -> VertexOutput {
    var sprite_d = sprite_data[i_idx];

    var local_position = vec2<f32>(0.0, 0.0);
    var local_uv = vec2<f32>(0.0, 0.0);

    switch (v_idx){
        case 0u: {local_position = vec2<f32>(-0.5, 0.5); local_uv = vec2<f32>(0.0, 0.0);}
        case 1u: {local_position = vec2<f32>(-0.5, -0.5); local_uv = vec2<f32>(0.0, 1.0);}
        case 2u: {local_position = vec2<f32>(0.5, 0.5); local_uv = vec2<f32>(1.0, 0.0);}
        default: {local_position = vec2<f32>(0.5, -0.5); local_uv = vec2<f32>(1.0, 1.0);}
    }

    let world_pos = (local_position * sprite_d.scale) + sprite_d.pos;

    var out: VertexOutput;

    out.clip_position= vec4<f32>(world_pos, 0.0, 1.0);

    out.uv = (local_uv * sprite_d.uv_scale) + sprite_d.uv_offset;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(sprite_texture, sprite_sampler, in.uv);

    if (color.a < 0.05){
        discard;
    }

    return color;
}
