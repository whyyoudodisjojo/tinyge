use image::{DynamicImage, GenericImageView};
use wgpu::{
    Extent3d, Origin3d, Queue, TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect,
    TextureView,
};

pub struct ResourceTexture {
    pub texture: Texture,
    pub view: TextureView,
    pub sz: Extent3d,
}

impl ResourceTexture {
    pub fn copy_image_data(&self, image: DynamicImage, queue: &Queue) {
        let rgba = image.to_rgba8();
        let dims = image.dimensions();

        let sz = Extent3d {
            width: dims.0,
            height: dims.1,
            depth_or_array_layers: 1,
        };

        assert_eq!(sz, self.sz);

        queue.write_texture(
            TexelCopyTextureInfo {
                aspect: TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
            },
            &rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dims.0),
                rows_per_image: Some(dims.1),
            },
            sz,
        );
    }
}
