use std::hash::{Hash, Hasher};

use wgpu::{
    Blas, BlasGeometrySizeDescriptors, Buffer, BufferDescriptor, BufferUsages, Device, Sampler,
    Texture, TextureDescriptor, TextureView, TextureViewDescriptor, Tlas,
    wgt::{CreateBlasDescriptor, CreateTlasDescriptor, SamplerDescriptor},
};

#[derive(Clone, Hash)]
pub struct TinyBuffer {
    pub inner: Option<Buffer>,
    pub size: u64,
    pub usages: BufferUsages,
}

impl TinyBuffer {
    pub fn new(sz: u64, usages: BufferUsages) -> Self {
        Self {
            inner: None,
            size: sz,
            usages,
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn usages(&self) -> BufferUsages {
        self.usages
    }

    pub fn bind(&mut self, buffer: Buffer) {
        self.inner = Some(buffer)
    }

    pub fn build(&mut self, device: &Device) {
        let buf = device.create_buffer(&BufferDescriptor {
            label: None,
            size: self.size,
            usage: self.usages,
            mapped_at_creation: false,
        });

        self.bind(buf);
    }

    pub fn raw(&self) -> &Buffer {
        self.inner.as_ref().unwrap()
    }
}

impl From<Buffer> for TinyBuffer {
    fn from(buffer: Buffer) -> Self {
        let size = buffer.size();
        let usages = buffer.usage();
        TinyBuffer {
            inner: Some(buffer),
            size,
            usages,
        }
    }
}

#[derive(Clone, Hash)]
pub struct TinyTexture<'a> {
    pub inner: Option<Texture>,
    pub view: Option<TextureView>,
    pub descriptor: TextureDescriptor<'a>,
}

impl<'a> TinyTexture<'a> {
    pub fn new(descriptor: TextureDescriptor<'a>) -> Self {
        Self {
            inner: None,
            view: None,
            descriptor,
        }
    }
    pub fn bind(&mut self, texture: Texture) {
        let texture_view = texture.create_view(&TextureViewDescriptor::default());
        self.inner = Some(texture);
        self.view = Some(texture_view);
    }

    pub fn build(&mut self, device: &Device) {
        let buf = device.create_texture(&self.descriptor);
        self.bind(buf);
    }

    pub fn raw(&self) -> &Texture {
        self.inner.as_ref().unwrap()
    }

    pub fn raw_view(&self) -> &TextureView {
        self.view.as_ref().unwrap()
    }
}

#[derive(Clone)]
pub struct TinyBlas<'a> {
    pub inner: Option<Blas>,
    pub descriptor: CreateBlasDescriptor<Option<&'a str>>,
    pub sz_descriptor: BlasGeometrySizeDescriptors,
}

impl<'a> Hash for TinyBlas<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<'a> TinyBlas<'a> {
    pub fn new(
        descriptor: CreateBlasDescriptor<Option<&'a str>>,
        size_desc: BlasGeometrySizeDescriptors,
    ) -> Self {
        Self {
            inner: None,
            descriptor,
            sz_descriptor: size_desc,
        }
    }
    pub fn bind(&mut self, blas: Blas) {
        self.inner = Some(blas)
    }

    pub fn build(&mut self, device: &Device) {
        let blas = device.create_blas(&self.descriptor, self.sz_descriptor.clone());
        self.bind(blas);
    }

    pub fn raw(&self) -> &Blas {
        self.inner.as_ref().unwrap()
    }
}

#[derive(Clone, Hash)]
pub struct TinyTlas<'a> {
    pub inner: Option<Tlas>,
    pub descriptor: CreateTlasDescriptor<Option<&'a str>>,
}

impl<'a> TinyTlas<'a> {
    pub fn new(descriptor: CreateTlasDescriptor<Option<&'a str>>) -> Self {
        Self {
            inner: None,
            descriptor,
        }
    }
    pub fn bind(&mut self, tlas: Tlas) {
        self.inner = Some(tlas)
    }

    pub fn build(&mut self, device: &Device) {
        let blas = device.create_tlas(&self.descriptor);
        self.bind(blas);
    }

    pub fn raw(&self) -> &Tlas {
        self.inner.as_ref().unwrap()
    }

    pub fn raw_mut(&mut self) -> &mut Tlas {
        self.inner.as_mut().unwrap()
    }
}

#[derive(Clone)]
pub struct TinySampler<'a> {
    pub inner: Option<Sampler>,
    pub descriptor: SamplerDescriptor<Option<&'a str>>,
}

impl<'a> Hash for TinySampler<'a> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl<'a> TinySampler<'a> {
    pub fn new(descriptor: SamplerDescriptor<Option<&'a str>>) -> Self {
        Self {
            inner: None,
            descriptor,
        }
    }
    pub fn bind(&mut self, sampler: Sampler) {
        self.inner = Some(sampler)
    }

    pub fn build(&mut self, device: &Device) {
        let sampler = device.create_sampler(&self.descriptor);
        self.bind(sampler);
    }

    pub fn raw(&self) -> &Sampler {
        self.inner.as_ref().unwrap()
    }
}
