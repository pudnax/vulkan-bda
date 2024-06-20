use core::slice;
use std::{alloc, collections::HashSet, ptr::NonNull, sync::Arc};

use crate::{instance::Instance, surface::Surface};

use anyhow::{Context, Result};
use ash::{
    ext, khr,
    prelude::VkResult,
    vk::{self, Handle},
};

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family: u32,
    pub device: Arc<ash::Device>,
    pub ext: Arc<DeviceExt>,
}

pub struct DeviceExt {
    pub descriptor_buffer: ext::descriptor_buffer::Device,
    pub dynamic_rendering: khr::dynamic_rendering::Device,
    pub shader_object: ext::shader_object::Device,
    pub host_memory: ext::external_memory_host::Device,
    pub desc_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT<'static>,
    min_host_pointer_alignment: u64,
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Device {
    pub fn new(instance: &Instance, surface: &Surface) -> Result<Self> {
        let required_device_extensions = [
            khr::swapchain::NAME,
            // these are all required for shader_object
            ext::shader_object::NAME,
            khr::dynamic_rendering::NAME,
            khr::depth_stencil_resolve::NAME,
            khr::create_renderpass2::NAME,
            khr::multiview::NAME,
            khr::synchronization2::NAME,
            khr::buffer_device_address::NAME,
            ext::external_memory_host::NAME,
            ext::descriptor_buffer::NAME,
        ];
        let required_device_extensions_set = HashSet::from(required_device_extensions);

        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let (pdevice, queue_family) = devices
            .into_iter()
            .filter_map(|device| {
                let extensions =
                    unsafe { instance.enumerate_device_extension_properties(device) }.ok()?;
                let extensions: HashSet<_> = extensions
                    .iter()
                    .map(|x| x.extension_name_as_c_str().unwrap())
                    .collect();
                let missing = required_device_extensions_set.difference(&extensions);
                if missing.count() != 0 {
                    return None;
                }

                let queue_properties =
                    unsafe { instance.get_physical_device_queue_family_properties(device) };
                let family_idx = queue_properties
                    .into_iter()
                    .enumerate()
                    .filter_map(|(family_idx, properties)| {
                        let family_idx = family_idx as u32;

                        let queue_support = properties
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER);
                        let surface_support =
                            surface.get_device_surface_support(device, family_idx);
                        (queue_support && surface_support).then_some(family_idx)
                    })
                    .next()?;

                Some((device, family_idx))
            })
            .next()
            .context("Failed to find suitable device.")?;

        let queue_priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&queue_priorities)];

        let required_device_extensions = required_device_extensions.map(|x| x.as_ptr());

        let mut feature_descriptor_buffer =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::default().descriptor_buffer(true);
        let mut feature_buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut feature_synchronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut feature_shader_object =
            vk::PhysicalDeviceShaderObjectFeaturesEXT::default().shader_object(true);
        let mut feature_dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);
        let default_features = vk::PhysicalDeviceFeatures::default();
        let device_info = vk::DeviceCreateInfo::default()
            .enabled_features(&default_features)
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&required_device_extensions)
            .push_next(&mut feature_descriptor_buffer)
            .push_next(&mut feature_buffer_device_address)
            .push_next(&mut feature_synchronization2)
            .push_next(&mut feature_shader_object)
            .push_next(&mut feature_dynamic_rendering);
        let device = unsafe { instance.instance.create_device(pdevice, &device_info, None) }?;

        fn fmt_size(n: u64) -> String {
            if n < 1_000_000 {
                format!("{:>3} B", n)
            } else if n < 1_000_000 {
                format!("{:>3} kB", n >> 10)
            } else if n < 1_000_000_000 {
                format!("{:>3} MB", n >> 20)
            } else {
                format!("{:>3} GB", n >> 30)
            }
        }
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };

        for mp in &memory_properties.memory_types[..memory_properties.memory_type_count as _] {
            if !mp.property_flags.is_empty() {
                println!("Memory: {:?}", mp.property_flags);
                let heap = memory_properties.memory_heaps[mp.heap_index as usize];
                println!(
                    "\tMemory Heap {}: Size {:?} | Type {:?}",
                    mp.heap_index,
                    fmt_size(heap.size),
                    heap.flags
                )
            }
        }
        let mut host_memory_properties =
            vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut descriptor_buffer_properties =
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut host_memory_properties)
            .push_next(&mut descriptor_buffer_properties);
        unsafe { instance.get_physical_device_properties2(pdevice, &mut props2) };

        let descriptor_buffer = ext::descriptor_buffer::Device::new(instance, &device);
        let host_memory = ext::external_memory_host::Device::new(instance, &device);
        let shader_object = ext::shader_object::Device::new(instance, &device);
        let dynamic_rendering = khr::dynamic_rendering::Device::new(instance, &device);

        let uniform_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER);
        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .bindings(slice::from_ref(&uniform_binding));
        let uniform_desc_layout = unsafe {
            device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None)?
        };
        let layout_size =
            unsafe { descriptor_buffer.get_descriptor_set_layout_size(uniform_desc_layout) };
        let layout_offset = unsafe {
            descriptor_buffer.get_descriptor_set_layout_binding_offset(uniform_desc_layout, 0)
        };

        let layout_size = align_to(
            layout_size,
            descriptor_buffer_properties.descriptor_buffer_offset_alignment,
        );

        let desc_set_buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .usage(
                        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                            | vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
                    )
                    .size(layout_size),
                None,
            )?
        };

        let requirements = unsafe { device.get_buffer_memory_requirements(desc_set_buffer) };
        let memory_type_index = find_memory_type_index(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("Failed to find suitable memory index for buffer memory");
        let mut alloc_flag =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut alloc_flag);
        let memory = unsafe { device.allocate_memory(&alloc_info, None) }?;
        unsafe { device.bind_buffer_memory(desc_set_buffer, memory, 0) }?;
        let address = unsafe {
            device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(desc_set_buffer),
            )
        };

        let desc_address_info = vk::DescriptorAddressInfoEXT::default()
            .format(vk::Format::UNDEFINED)
            .address(address)
            .range(size_of::<[[f32; 3]; 3]>() as _);
        let desc_data = vk::DescriptorDataEXT {
            p_uniform_buffer: &desc_address_info,
        };
        let mut x = [0; 8];
        unsafe {
            descriptor_buffer.get_descriptor(
                &vk::DescriptorGetInfoEXT::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .data(desc_data),
                &mut x,
            )
        };

        Ok(Self {
            physical_device: pdevice,
            queue_family,
            memory_properties,
            device: Arc::new(device),
            ext: Arc::new(DeviceExt {
                dynamic_rendering,
                descriptor_buffer,
                shader_object,
                host_memory,
                desc_buffer_properties: descriptor_buffer_properties,
                min_host_pointer_alignment: host_memory_properties
                    .min_imported_host_pointer_alignment,
            }),
        })
    }

    fn get_host_memory_properties(
        &self,
        ptr: *mut u8,
    ) -> VkResult<vk::MemoryHostPointerPropertiesEXT> {
        let mut mem_properties = vk::MemoryHostPointerPropertiesEXT::default();
        let fp = self.ext.host_memory.fp();
        let result = unsafe {
            (fp.get_memory_host_pointer_properties_ext)(
                self.ext.host_memory.device(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                ptr.cast(),
                &mut mem_properties as _,
            )
        };
        match result {
            vk::Result::SUCCESS => Ok(mem_properties),
            _ => Err(result),
        }
    }

    pub fn get_buffer_address(&self, buffer: &Buffer) -> u64 {
        unsafe {
            self.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer),
            )
        }
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_prop_flags: vk::MemoryPropertyFlags,
    ) -> VkResult<Buffer> {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                None,
            )?
        };
        let requirements = unsafe { self.get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type_index(
            &self.memory_properties,
            requirements.memory_type_bits,
            memory_prop_flags,
        )
        .expect("Failed to find suitable memory index for buffer memory");
        let mut alloc_flag =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut alloc_flag);
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }?;
        unsafe { self.bind_buffer_memory(buffer, memory, 0) }?;
        let address = unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        Ok(Buffer {
            address,
            buffer,
            memory,
            device: self.device.clone(),
        })
    }

    pub fn create_host_buffer<T>(&self, usage: vk::BufferUsageFlags) -> Result<HostBuffer<T>> {
        let size = size_of::<T>() as u64;
        let alignment = self.ext.min_host_pointer_alignment;
        let ptr = unsafe {
            NonNull::new(alloc::alloc(alloc::Layout::from_size_align(
                size as usize,
                alignment as usize,
            )?))
            .context("Failed to allocate pointer for host memory")?
            .as_ptr()
        };
        let ptr_aligned = ((ptr as usize) & !(alignment as usize - 1)) as *mut u8;
        let data_offset = ptr as usize & (alignment as usize - 1);
        let data_size = align_to(size + data_offset as u64, alignment);

        let host_memory_properties = self.get_host_memory_properties(ptr)?;
        let mut import_memory_info = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(ptr_aligned.cast());
        let memory_type_index = find_memory_type_index(
            &self.memory_properties,
            host_memory_properties.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .context("Failed to find suitable memory index for host memory")?;
        let mut alloc_flag =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let memory = unsafe {
            self.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(data_size)
                    .memory_type_index(memory_type_index)
                    .push_next(&mut import_memory_info)
                    .push_next(&mut alloc_flag),
                None,
            )?
        };

        let mut host_buffer_create_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size_of::<T>() as _)
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                    .push_next(&mut host_buffer_create_info),
                None,
            )?
        };
        unsafe { self.bind_buffer_memory(buffer, memory, 0) }?;
        let address = unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        let ptr = unsafe { Box::from_raw(ptr.byte_add(data_offset).cast()) };
        Ok(HostBuffer {
            address,
            buffer,
            memory,
            ptr,
            device: self.device.clone(),
        })
    }

    pub fn create_render_shader<P: ?Sized + AsRef<std::path::Path>>(
        &self,
        compiler: &mut shaderc::Compiler,
        vs_glsl_path: &P,
        fs_glsl_path: &P,
        push_constant_ranges: &[vk::PushConstantRange],
        descriptor_set_layout: &[vk::DescriptorSetLayout],
    ) -> Result<(vk::ShaderEXT, vk::ShaderEXT)> {
        let mut options =
            shaderc::CompileOptions::new().context("Failed to create shader compiler options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        options.set_generate_debug_info();

        let vert_src = std::fs::read_to_string(vs_glsl_path)?;
        let vert = compiler.compile_into_spirv(
            &vert_src,
            shaderc::ShaderKind::Vertex,
            &vs_glsl_path.as_ref().to_string_lossy(),
            "main",
            Some(&options),
        )?;

        let frag_src = std::fs::read_to_string(fs_glsl_path)?;
        let frag = compiler.compile_into_spirv(
            &frag_src,
            shaderc::ShaderKind::Fragment,
            &fs_glsl_path.as_ref().to_string_lossy(),
            "main",
            Some(&options),
        )?;

        let shader_infos = [
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::VERTEX)
                .next_stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(vert.as_binary_u8())
                .name(c"main")
                .push_constant_ranges(push_constant_ranges)
                .set_layouts(descriptor_set_layout),
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .next_stage(vk::ShaderStageFlags::empty())
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(frag.as_binary_u8())
                .name(c"main")
                .push_constant_ranges(push_constant_ranges)
                .set_layouts(descriptor_set_layout),
        ];
        match unsafe { self.ext.shader_object.create_shaders(&shader_infos, None) } {
            Ok(ret) => Ok((ret[0], ret[1])),

            Err((ret, err)) => {
                if ret[0].is_null() {
                    panic!("Failed to compile vertex shader: {err}")
                } else if ret[1].is_null() {
                    panic!("Failed to compile fragment shader: {err}")
                } else {
                    panic!("Shader compilation failed: {err}")
                }
            }
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

pub fn align_to(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

pub fn find_memory_type_index(
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_type_bits != 0 && (memory_type.property_flags & flags) == flags
        })
        .map(|(index, _memory_type)| index as _)
}

pub struct Buffer {
    pub address: u64,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    device: Arc<ash::Device>,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

pub struct HostBuffer<T> {
    pub address: u64,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: Box<T>,
    device: Arc<ash::Device>,
}

impl<T> std::ops::Deref for HostBuffer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

impl<T> std::ops::DerefMut for HostBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.ptr
    }
}

impl<T> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
