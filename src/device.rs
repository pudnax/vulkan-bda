use std::{collections::HashSet, sync::Arc};

use crate::{instance::Instance, surface::Surface};

use anyhow::{Context, Result};
use ash::{
    ext::{self},
    khr, vk,
};

pub struct Device {
    // TODO: remove
    pub shader_device: ext::shader_object::Device,
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
            shader_device: shader_object.clone(),
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
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
