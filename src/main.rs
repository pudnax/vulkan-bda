use core::slice;
use std::{f32::consts::TAU, time::Instant};

use anyhow::{Context, Result};
use ash::{khr, vk};
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use self::{
    device::{align_to, Buffer, Device, HostBuffer},
    instance::Instance,
    surface::Surface,
    swapchain::Swapchain,
};

mod device;
mod instance;
mod surface;
mod swapchain;

struct AppInit {
    vs: vk::ShaderEXT,
    fs: vk::ShaderEXT,
    pipeline_layout: vk::PipelineLayout,
    window: Window,
    queue: vk::Queue,

    time: Instant,
    pc_host_buffer: HostBuffer<[f32; 4]>,
    un_host_buffer: HostBuffer<[[f32; 4]; 3]>,
    // un_host_buffer: Buffer,
    un_desc_buffer: Buffer,
    mapped: *mut u8,

    compiler: shaderc::Compiler,
    swapchain: Swapchain,
    surface: Surface,
    device: Device,
    instance: Instance,
}

impl Drop for AppInit {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
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
        let swapchain = Swapchain::new(&device, &surface, swapchain_loader)?;

        let mut compiler = shaderc::Compiler::new().context("Failed to create shader compiler")?;

        let pc_host_buffer = device.create_host_buffer(vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        let mut un_host_buffer = device.create_host_buffer(vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        // let un_host_buffer = device.create_buffer(
        //     size_of::<[[f32; 3]; 3]>() as _,
        //     vk::BufferUsageFlags::UNIFORM_BUFFER,
        //     vk::MemoryPropertyFlags::HOST_VISIBLE,
        // )?;

        let uniform_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER);
        let uniform_desc_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .flags(vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT)
                    .bindings(slice::from_ref(&uniform_binding)),
                None,
            )?
        };
        let buffer_offset_alignment = device
            .ext
            .desc_buffer_properties
            .descriptor_buffer_offset_alignment;
        let layout_size = unsafe {
            let size = device
                .ext
                .descriptor_buffer
                .get_descriptor_set_layout_size(uniform_desc_layout);
            align_to(size, buffer_offset_alignment)
        };
        let layout_offset = unsafe {
            device
                .ext
                .descriptor_buffer
                .get_descriptor_set_layout_binding_offset(uniform_desc_layout, 0)
                as usize
        };
        dbg!(&layout_offset);
        dbg!(&layout_size);

        let un_desc_buffer = device.create_buffer(
            layout_size,
            vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mapped = unsafe {
            device.map_memory(
                un_desc_buffer.memory,
                0,
                layout_size,
                vk::MemoryMapFlags::empty(),
            )?
        };

        let desc_address_info = vk::DescriptorAddressInfoEXT::default()
            // .format(vk::Format::UNDEFINED)
            .address(un_host_buffer.address)
            .range(size_of::<[[f32; 4]; 3]>() as _);
        let desc_data = vk::DescriptorDataEXT {
            p_uniform_buffer: &desc_address_info,
        };
        let un_buffer_size = device
            .ext
            .desc_buffer_properties
            .uniform_buffer_descriptor_size;
        dbg!(&un_buffer_size);

        // println!("Origignal:\t{:p}", un_host_buffer.ptr);
        // let buffer_ptr = bytemuck::bytes_of_mut(&mut *un_host_buffer.ptr);
        // println!("Before:\t{:p}", buffer_ptr);
        // let x = &mut buffer_ptr[layout_offset..][..un_buffer_size];
        // println!("After:\t{:p}", x);
        let x = unsafe { std::slice::from_raw_parts_mut(mapped.cast(), layout_size as usize) };
        let x = &mut x[layout_offset..][..un_buffer_size];
        unsafe {
            device.ext.descriptor_buffer.get_descriptor(
                &vk::DescriptorGetInfoEXT::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .data(desc_data),
                x,
            )
        };

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(size_of::<u64>() as _);
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(slice::from_ref(&uniform_desc_layout))
                    .push_constant_ranges(slice::from_ref(&push_constant_range)),
                None,
            )?
        };
        let (vs, fs) = device.create_render_shader(
            &mut compiler,
            "src/trig.vert.glsl",
            "src/trig.frag.glsl",
            &[push_constant_range],
            &[uniform_desc_layout],
        )?;

        Ok(Self {
            vs,
            fs,
            window,
            queue,

            time: Instant::now(),

            pc_host_buffer,
            un_host_buffer,
            pipeline_layout,
            un_desc_buffer,
            mapped: mapped.cast(),

            compiler,
            swapchain,
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
                let t = self.time.elapsed().as_secs_f32();
                let (c, s) = (t.cos(), t.sin());
                self.pc_host_buffer.copy_from_slice(&[c, -s, s, c]);

                let cos_palette = |t: f32| {
                    let a = [0.5, 0.5, 0.5, 0.];
                    let b = [0.5, 0.5, 0.5, 0.];
                    let c = [1., 1., 0.5, 0.];
                    let d = [0.8, 0.9, 0.3, 0.];
                    std::array::from_fn(|i| a[i] + b[i] * f32::cos(TAU * (c[i] * t + d[i])))
                };
                for (i, col) in self.un_host_buffer.iter_mut().enumerate() {
                    *col = cos_palette(t + 0.2 * i as f32)
                }

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
                    [0., 0.025, 0.025, 1.0],
                );
                unsafe {
                    frame.ext.descriptor_buffer.cmd_bind_descriptor_buffers(
                        frame.frame.command_buffer,
                        &[vk::DescriptorBufferBindingInfoEXT::default()
                            .usage(vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT)
                            .address(self.un_desc_buffer.address)],
                    )
                };
                unsafe {
                    frame
                        .ext
                        .descriptor_buffer
                        .cmd_set_descriptor_buffer_offsets(
                            frame.frame.command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[0],
                            &[0],
                        )
                };
                frame.push_constant(
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    &self.pc_host_buffer.address,
                );
                frame.bind_vs_fs(self.vs, self.fs);

                frame.draw(3, 0);
                frame.end_rendering();
                match self.swapchain.submit_image(&self.queue, frame) {
                    Ok(_) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain(),
                    Err(e) => panic!("error: {e}\n"),
                }

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
