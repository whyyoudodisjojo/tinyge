struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@group(0) @binding(0) var<uniform> time: f32;

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let angle = time * 0.5;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    
    var rotated_pos = model.position;
    rotated_pos.x = model.position.x * cos_a - model.position.y * sin_a;
    rotated_pos.y = model.position.x * sin_a + model.position.y * cos_a;
    
    out.clip_position = vec4<f32>(rotated_pos, 1.0);
    out.color = model.color;
    out.uv = vec2<f32>(model.position.x, model.position.y);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let t = time * 0.3;
    
    let gradient = sin(in.uv.x * 3.0 + t) * cos(in.uv.y * 3.0 + t * 0.7);
    
    let rainbow_r = 0.5 + 0.5 * sin(t + 0.0);
    let rainbow_g = 0.5 + 0.5 * sin(t + 2.094);
    let rainbow_b = 0.5 + 0.5 * sin(t + 4.189);
    
    let final_color = in.color * 0.6 + vec3<f32>(rainbow_r, rainbow_g, rainbow_b) * 0.4;
    
    let pulse = 0.85 + 0.15 * sin(t * 2.0);
    
    return vec4<f32>(final_color * pulse, 1.0);
}
