use crate::{device::Device, surface::Surface};

use anyhow::Result;
use ash::{khr, vk, Entry};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct Instance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
}

impl std::ops::Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Instance {
    pub fn new(display_handle: Option<&impl HasDisplayHandle>) -> Result<Self> {
        let entry = unsafe { Entry::load() }?;
        let layers = [
            #[cfg(debug_assertions)]
            c"VK_LAYER_KHRONOS_validation".as_ptr(),
        ];
        let mut extensions = vec![
            khr::surface::NAME.as_ptr(),
            khr::display::NAME.as_ptr(),
            khr::get_physical_device_properties2::NAME.as_ptr(),
        ];
        if let Some(handle) = display_handle {
            extensions.extend(ash_window::enumerate_required_extensions(
                handle.display_handle()?.as_raw(),
            )?);
        }

        let appinfo = vk::ApplicationInfo::default()
            .application_name(c"Modern Vulkan")
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(vk::InstanceCreateFlags::default())
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;
        Ok(Self { entry, instance })
    }

    pub fn create_device(&self, surface: &Surface) -> Result<Device> {
        Device::new(self, surface)
    }

    pub fn create_surface(
        &self,
        handle: &(impl HasDisplayHandle + HasWindowHandle),
    ) -> Result<Surface> {
        Surface::new(&self.entry, &self.instance, handle)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}
