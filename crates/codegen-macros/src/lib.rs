mod buffer_struct_deser;
mod shader;
mod wgsl_structs;

use proc_macro::TokenStream;

use crate::{
    buffer_struct_deser::buffer_struct_deser, shader::shader_inner,
    wgsl_structs::derive_into_wgsl_struct_inner,
};

#[proc_macro_derive(IntoWgslStruct, attributes(codegen))]
pub fn derive_into_wgsl_struct(item: TokenStream) -> TokenStream {
    derive_into_wgsl_struct_inner(item)
}

#[proc_macro_attribute]
pub fn shader(attr: TokenStream, item: TokenStream) -> TokenStream {
    shader_inner(attr, item)
}

#[proc_macro_derive(IntoBufferStruct, attributes(vertex, index, resource))]
pub fn into_buffer_struct(item: TokenStream) -> TokenStream {
    buffer_struct_deser(item)
}
