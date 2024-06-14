use std::{collections::HashSet, sync::Arc};

use crate::{instance::Instance, surface::Surface};

use anyhow::{Context, Result};
use ash::{
    ext, khr,
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
    pub dynamic_rendering: khr::dynamic_rendering::Device,
    pub shader_object: ext::shader_object::Device,
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
            .push_next(&mut feature_synchronization2)
            .push_next(&mut feature_shader_object)
            .push_next(&mut feature_dynamic_rendering);
        let device = unsafe { instance.instance.create_device(pdevice, &device_info, None) }?;

        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let shader_object = ash::ext::shader_object::Device::new(instance, &device);
        let dynamic_rendering = khr::dynamic_rendering::Device::new(instance, &device);

        Ok(Self {
            physical_device: pdevice,
            queue_family,
            memory_properties,
            device: Arc::new(device),
            ext: Arc::new(DeviceExt {
                dynamic_rendering,
                shader_object,
            }),
        })
    }

    pub fn create_graphics_shader<P: ?Sized + AsRef<std::path::Path>>(
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
                .push_constant_ranges(&push_constant_ranges)
                .set_layouts(&descriptor_set_layout),
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .next_stage(vk::ShaderStageFlags::empty())
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(frag.as_binary_u8())
                .name(c"main")
                .push_constant_ranges(&push_constant_ranges)
                .set_layouts(&descriptor_set_layout),
        ];
        match unsafe { self.ext.shader_object.create_shaders(&shader_infos, None) } {
            Ok(ret) => Ok((ret[0], ret[1])),

            Err((ret, err)) => {
                if ret[0].is_null() {
                    panic!("\n vertex shader failed to compile\n{err}\n")
                } else if ret[1].is_null() {
                    panic!("\n fragment shader failed to compile\n{err}\n")
                } else {
                    panic!("\n shader compilation failed\n{err}\n")
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
