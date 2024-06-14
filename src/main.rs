use anyhow::Result;
use ash::{
    ext, khr,
    vk::{self, Handle},
};
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use self::{device::Device, instance::Instance, surface::Surface, swapchain::Swapchain};

mod device;
mod instance;
mod surface;
mod swapchain;

struct AppInit {
    vs: vk::ShaderEXT,
    fs: vk::ShaderEXT,
    window: Window,
    queue: vk::Queue,

    swapchain: Swapchain,
    surface: Surface,
    device: Device,
    instance: Instance,
}

impl Drop for AppInit {
    fn drop(&mut self) {
        unsafe {
            self.device.ext.shader_object.destroy_shader(self.vs, None);
            self.device.ext.shader_object.destroy_shader(self.fs, None);
        }
    }
}

impl AppInit {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_attributes: WindowAttributes,
    ) -> Result<Self> {
        let window = event_loop.create_window(window_attributes)?;

        let instance = Instance::new(Some(&window))?;
        let surface = instance.create_surface(&window)?;
        let device = instance.create_device(&surface)?;
        let queue = unsafe { device.get_device_queue(device.queue_family, 0) };

        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let swapchain2 = Swapchain::new(&device, &surface, swapchain_loader)?;

        let (vs, fs) = Self::load_glsl_vs_fs(
            &device.shader_device,
            "src/trig.vert.glsl",
            "src/trig.frag.glsl",
            &[],
            &[], // &push_constant_ranges,
                 // &set_layouts,
        );

        Ok(Self {
            vs,
            fs,
            window,
            queue,

            swapchain: swapchain2,
            surface,
            device,
            instance,
        })
    }

    fn recreate_swapchain(&mut self) {
        self.swapchain
            .recreate(&self.device, &self.surface)
            .expect("Failed to recreate swapchain")
    }

    pub fn load_glsl_vs_fs<P: ?Sized + AsRef<std::path::Path>>(
        device: &ext::shader_object::Device,
        vs_glsl_path: &P,
        fs_glsl_path: &P,
        push_constant_ranges: &[vk::PushConstantRange],
        descriptor_set_layout: &[vk::DescriptorSetLayout],
    ) -> (vk::ShaderEXT, vk::ShaderEXT) {
        use shaderc;
        let compiler = shaderc::Compiler::new().unwrap();
        let options = shaderc::CompileOptions::new().unwrap();

        let vert_src = std::fs::read_to_string(vs_glsl_path).expect("could not read vertex shader");
        let vert = compiler
            .compile_into_spirv(
                &vert_src,
                shaderc::ShaderKind::Vertex,
                &vs_glsl_path.as_ref().to_string_lossy(),
                "main",
                Some(&options),
            )
            .expect("vert shader failed to compile");

        let frag_src =
            std::fs::read_to_string(fs_glsl_path).expect("could not read fragment shader");
        let frag = compiler
            .compile_into_spirv(
                &frag_src,
                shaderc::ShaderKind::Fragment,
                &fs_glsl_path.as_ref().to_string_lossy(),
                "main",
                Some(&options),
            )
            .expect("vert shader failed to compile");
        Self::load_spirv_vs_fs(
            device,
            vert.as_binary_u8(),
            frag.as_binary_u8(),
            push_constant_ranges,
            descriptor_set_layout,
        )
    }

    pub fn load_spirv_vs_fs(
        device: &ext::shader_object::Device,
        vs_spv: &[u8],
        fs_spv: &[u8],
        push_constant_ranges: &[vk::PushConstantRange],
        descriptor_set_layout: &[vk::DescriptorSetLayout],
    ) -> (vk::ShaderEXT, vk::ShaderEXT) {
        let shader_infos = [
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::VERTEX)
                .next_stage(vk::ShaderStageFlags::FRAGMENT)
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(vs_spv)
                .name(c"main")
                .push_constant_ranges(&push_constant_ranges)
                .set_layouts(&descriptor_set_layout),
            vk::ShaderCreateInfoEXT::default()
                .flags(vk::ShaderCreateFlagsEXT::LINK_STAGE)
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .next_stage(vk::ShaderStageFlags::empty())
                .code_type(vk::ShaderCodeTypeEXT::SPIRV)
                .code(fs_spv)
                .name(c"main")
                .push_constant_ranges(&push_constant_ranges)
                .set_layouts(&descriptor_set_layout),
        ];
        match unsafe { device.create_shaders(&shader_infos, None) } {
            Ok(ret) => (ret[0], ret[1]),
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

    pub fn transition_image(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        from: vk::ImageLayout,
        to: vk::ImageLayout,
    ) {
        let subrange: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = [vk::ImageMemoryBarrier::default()
            .image(image)
            .old_layout(from)
            .new_layout(to)
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .subresource_range(subrange)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::BY_REGION,
                &[],
                &[],
                &barrier,
            )
        };
    }
}

impl ApplicationHandler for AppInit {
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let (vs, fs) = (self.vs, self.fs);

                let mut frame = match self.swapchain.acquire_next_image() {
                    Ok(frame) => frame,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.recreate_swapchain();
                        self.window.request_redraw();
                        return;
                    }
                    Err(e) => panic!("error: {e}\n"),
                };
                frame.begin_rendering(
                    self.swapchain.get_current_image_view(),
                    [
                        0.,
                        // (0x32 as f32 / 0xFF as f32).powf(2.2),
                        (0x30 as f32 / 0xFF as f32).powf(2.2),
                        (0x2f as f32 / 0xFF as f32).powf(2.2),
                        1.0,
                    ],
                );
                frame.bind_vs_fs(vs, fs);

                frame.draw(3, 0);
                frame.end_rendering();
                match self.swapchain.submit_image(&self.queue, frame) {
                    Ok(_) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain(),
                    Err(e) => panic!("error: {e}\n"),
                }

                // self.window.pre_present_notify();
                self.window.request_redraw();
            }
            _ => {}
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = unsafe { self.device.device_wait_idle() };
    }

    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        panic!("On native platforms `resumed` can be called only once.")
    }
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::new()?;

    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
enum App {
    #[default]
    Uninitialized,
    Init(AppInit),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = WindowAttributes::default();
        match self {
            Self::Uninitialized => {
                *self = Self::Init(
                    AppInit::new(event_loop, window_attributes)
                        .expect("Failed to create application"),
                )
            }
            Self::Init(_) => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Self::Init(app) = self {
            app.window_event(event_loop, window_id, event);
        }
    }

    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        if let Self::Init(app) = self {
            app.new_events(event_loop, cause);
        }
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: ()) {
        if let Self::Init(app) = self {
            app.user_event(event_loop, event)
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Self::Init(app) = self {
            app.device_event(event_loop, device_id, event)
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.about_to_wait(event_loop)
        }
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.suspended(event_loop)
        }
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.exiting(event_loop)
        }
    }

    fn memory_warning(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Self::Init(app) = self {
            app.memory_warning(event_loop)
        }
    }
}
