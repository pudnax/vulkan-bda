use anyhow::Result;
use ash::{
    ext, khr,
    vk::{self, Extent2D, Handle},
};
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

use self::{device::Device, instance::Instance, surface::Surface};

mod device;
mod instance;
mod surface;

struct AppInit {
    vs: vk::ShaderEXT,
    fs: vk::ShaderEXT,
    window: Window,
    queue: vk::Queue,
    surface_format: vk::SurfaceFormatKHR,
    swapchain: vk::SwapchainKHR,
    swapchain_extent: Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    ready_to_submit: vk::Semaphore,
    ready_to_record: vk::Fence,
    ready_to_present: vk::Semaphore,
    khr_swapchain: khr::swapchain::Device,
    khr_dynamic_rendering: khr::dynamic_rendering::Device,

    surface: Surface,
    device: Device,
    instance: Instance,
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
        let khr_dynamic_rendering = khr::dynamic_rendering::Device::new(&instance, &device);

        let khr_swapchain = khr::swapchain::Device::new(&instance, &device);
        let surface_format = vk::SurfaceFormatKHR {
            format: vk::Format::B8G8R8A8_SRGB,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        };
        let (swapchain, swapchain_images, swapchain_views, swapchain_extent) =
            Self::create_swapchain(&device, &khr_swapchain, &surface, surface_format);

        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(device.queue_family);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }.unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap()[0];

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let ready_to_submit = unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap();
        let ready_to_present = unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let ready_to_record = unsafe { device.create_fence(&fence_info, None) }.unwrap();

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
            surface_format,
            swapchain,
            swapchain_extent,
            swapchain_images,
            swapchain_views,
            command_pool,
            command_buffer,
            ready_to_submit,
            ready_to_record,
            ready_to_present,
            khr_swapchain,
            khr_dynamic_rendering,

            surface,
            device,
            instance,
        })
    }

    fn create_swapchain(
        device: &Device,
        khr_swapchain: &khr::swapchain::Device,
        surface: &Surface,
        surface_format: vk::SurfaceFormatKHR,
    ) -> (
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        Extent2D,
    ) {
        let default_size = Extent2D {
            width: 1280,
            height: 720,
        }; // TODO: derive from Display
        let capabilities = surface.get_device_capabilities(device);
        let swapchain_extent = match capabilities.current_extent {
            Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            } => {
                let min = capabilities.min_image_extent;
                let max = capabilities.max_image_extent;
                vk::Extent2D {
                    width: default_size.width.clamp(min.width, max.width),
                    height: default_size.height.clamp(min.height, max.height),
                }
            }
            x => x,
        };
        let swapchain_info = vk::SwapchainCreateInfoKHR::default()
            .surface(**surface)
            .min_image_count(
                capabilities
                    .max_image_count
                    .min(capabilities.min_image_count + 1),
            )
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE);
        let swapchain = unsafe { khr_swapchain.create_swapchain(&swapchain_info, None) }.unwrap();
        let swapchain_images = unsafe { khr_swapchain.get_swapchain_images(swapchain) }.unwrap();
        let swapchain_views: Vec<_> = swapchain_images
            .iter()
            .map(|img| {
                let info = vk::ImageViewCreateInfo::default()
                    .image(*img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { device.create_image_view(&info, None) }.unwrap()
            })
            .collect();

        (
            swapchain,
            swapchain_images,
            swapchain_views,
            swapchain_extent,
        )
    }

    fn destroy_swapchain(&self) {
        // Note: swapchain images are owned by the the swapchain, so we only have to free the views
        for view in self.swapchain_views.iter() {
            unsafe { self.device.destroy_image_view(*view, None) };
        }
        unsafe { self.khr_swapchain.destroy_swapchain(self.swapchain, None) };
    }
    fn recreate_swapchain(&mut self) {
        self.destroy_swapchain();
        let (swapchain, swapchain_images, swapchain_views, swapchain_extent) =
            Self::create_swapchain(
                &self.device,
                &self.khr_swapchain,
                &self.surface,
                self.surface_format,
            );
        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_views = swapchain_views;
        self.swapchain_extent = swapchain_extent;
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

    pub fn wait_and_begin_frame(&mut self) -> Frame {
        Frame::new(self)
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

pub struct Frame<'a> {
    renderer: &'a mut AppInit,
    swap_idx: u32,
    dynamic_state_flags: DynamicStateFlags,
}
impl<'a> Frame<'a> {
    fn new(renderer: &'a mut AppInit) -> Self {
        // Synchronisation
        // three primitives:
        //  - ready_to_record:  signaled by VkQueueSubmit, awaited by host before vkAcquireNextImageKHR
        //  - ready_to_submit:  signaled by vkAcquireNextImageKHR, awaited by vkQueueSubmit
        //  - ready_to_present: signaled by vkQueueSubmit, awaited by vkQueuePresentKHR
        unsafe {
            renderer
                .device
                .wait_for_fences(&[renderer.ready_to_record], true, u64::MAX)
        }
        .unwrap();
        unsafe { renderer.device.reset_fences(&[renderer.ready_to_record]) }.unwrap();

        let swap_idx = loop {
            match unsafe {
                renderer.khr_swapchain.acquire_next_image(
                    renderer.swapchain,
                    u64::MAX,
                    renderer.ready_to_submit,
                    vk::Fence::null(),
                )
            } {
                Ok((swap_idx, false)) => break swap_idx,
                Ok((_, true)) => {
                    println!("resize! (acquire_next_image suboptimal)");
                    renderer.recreate_swapchain();
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    println!("resize! (acquire_next_image out of date)");
                    renderer.recreate_swapchain();
                }
                Err(e) => {
                    panic!("error: {e}\n");
                }
            };
        };

        // begin command buffer
        unsafe {
            renderer.device.reset_command_buffer(
                renderer.command_buffer,
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .unwrap();
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            renderer
                .device
                .begin_command_buffer(renderer.command_buffer, &begin_info)
        }
        .unwrap();

        // transition swapchain image from present-optimal to render-optimal
        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_memory_barriers = [vk::ImageMemoryBarrier::default()
            .image(renderer.swapchain_images[swap_idx as usize])
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .subresource_range(range)];
        unsafe {
            renderer.device.cmd_pipeline_barrier(
                renderer.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, // (changed from TOP_OF_PIPE)
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barriers,
            )
        };

        let dynamic_state_flags = DynamicStateFlags::empty();
        Self {
            renderer,
            swap_idx,
            dynamic_state_flags,
        }
    }

    pub fn begin_rendering(&self, color: [f32; 4]) {
        // begin rendering
        let mut clear_color_value = vk::ClearColorValue::default();
        clear_color_value.float32 = color;
        let mut clear_color = vk::ClearValue::default();
        clear_color.color = clear_color_value;
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.renderer.swapchain_views[self.swap_idx as usize])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_image_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.renderer.swapchain_extent.into())
            .layer_count(1)
            .color_attachments(&color_attachments);
        unsafe {
            self.renderer
                .khr_dynamic_rendering
                .cmd_begin_rendering(self.renderer.command_buffer, &rendering_info)
        };
    }

    pub fn set_viewports(&mut self, viewports: &[vk::Viewport]) {
        self.dynamic_state_flags |= DynamicStateFlags::VIEWPORTS;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_viewport_with_count(self.renderer.command_buffer, &viewports)
        };
    }
    pub fn set_scissors(&mut self, scissors: &[vk::Rect2D]) {
        self.dynamic_state_flags |= DynamicStateFlags::SCISSORS;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_scissor_with_count(self.renderer.command_buffer, &scissors)
        };
    }
    pub fn set_polygon_mode(&mut self, mode: vk::PolygonMode) {
        self.dynamic_state_flags |= DynamicStateFlags::POLYGON_MODE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_polygon_mode(self.renderer.command_buffer, mode)
        };
    }
    pub fn set_primitive_topology(&mut self, topology: vk::PrimitiveTopology) {
        self.dynamic_state_flags |= DynamicStateFlags::PRIMITIVE_TOPOLOGY;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_primitive_topology(self.renderer.command_buffer, topology)
        };
    }
    pub fn set_primitive_restart_enable(&mut self, enabled: bool) {
        self.dynamic_state_flags |= DynamicStateFlags::PRIMITIVE_RESTART_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_primitive_restart_enable(self.renderer.command_buffer, enabled)
        };
    }
    pub fn set_depth_test_enable(&mut self, enabled: bool) {
        self.dynamic_state_flags |= DynamicStateFlags::DEPTH_TEST_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_depth_test_enable(self.renderer.command_buffer, enabled)
        };
    }
    pub fn set_depth_write_enable(&mut self, enabled: bool) {
        self.dynamic_state_flags |= DynamicStateFlags::DEPTH_WRITE_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_depth_write_enable(self.renderer.command_buffer, enabled)
        };
    }
    pub fn set_depth_bias_enable(&mut self, enabled: bool) {
        self.dynamic_state_flags |= DynamicStateFlags::DEPTH_BIAS_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_depth_bias_enable(self.renderer.command_buffer, enabled)
        };
    }
    pub fn set_stencil_test_enable(&mut self, enabled: bool) {
        self.dynamic_state_flags |= DynamicStateFlags::STENCIL_TEST_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_stencil_test_enable(self.renderer.command_buffer, enabled)
        };
    }
    pub fn set_rasterizer_discard_enable(&mut self, enabled: bool) {
        self.dynamic_state_flags |= DynamicStateFlags::RASTERIZER_DISCARD_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_rasterizer_discard_enable(self.renderer.command_buffer, enabled)
        };
    }
    pub fn set_rasterization_samples(&mut self, sample_count_flags: vk::SampleCountFlags) {
        self.dynamic_state_flags |= DynamicStateFlags::RASTERIZATION_SAMPLES;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_rasterization_samples(self.renderer.command_buffer, sample_count_flags)
        };
    }
    pub fn set_sample_mask(
        &mut self,
        samples: vk::SampleCountFlags,
        sample_mask: &[vk::SampleMask],
    ) {
        self.dynamic_state_flags |= DynamicStateFlags::SAMPLE_MASK;
        unsafe {
            self.renderer.device.shader_device.cmd_set_sample_mask(
                self.renderer.command_buffer,
                samples,
                sample_mask,
            )
        };
    }
    pub fn set_alpha_to_coverage_enable(&mut self, enable: bool) {
        self.dynamic_state_flags |= DynamicStateFlags::ALPHA_TO_COVERAGE_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_alpha_to_coverage_enable(self.renderer.command_buffer, enable)
        };
    }
    pub fn set_front_face(&mut self, front_face: vk::FrontFace) {
        self.dynamic_state_flags |= DynamicStateFlags::FRONT_FACE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_front_face(self.renderer.command_buffer, front_face);
        }
    }
    pub fn set_cull_mode(&mut self, cullmode: vk::CullModeFlags) {
        self.dynamic_state_flags |= DynamicStateFlags::CULL_MODE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_cull_mode(self.renderer.command_buffer, cullmode)
        };
    }
    pub fn set_color_blend_enable(&mut self, enables: &[u32]) {
        self.dynamic_state_flags |= DynamicStateFlags::COLOR_BLEND_ENABLE;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_color_blend_enable(self.renderer.command_buffer, 0, &enables)
        };
    }
    pub fn set_color_blend_equation(&mut self, equations: &[vk::ColorBlendEquationEXT]) {
        self.dynamic_state_flags |= DynamicStateFlags::COLOR_BLEND_EQUATION;
        unsafe {
            self.renderer
                .device
                .shader_device
                .cmd_set_color_blend_equation(self.renderer.command_buffer, 0, &equations)
        };
    }
    pub fn set_color_write_mask(&mut self, write_masks: &[vk::ColorComponentFlags]) {
        self.dynamic_state_flags |= DynamicStateFlags::COLOR_WRITE_MASK;
        unsafe {
            self.renderer.device.shader_device.cmd_set_color_write_mask(
                self.renderer.command_buffer,
                0,
                &write_masks,
            )
        };
    }

    // vulkan requires us to set these things before rendering
    #[rustfmt::skip]
    fn apply_unset_defaults(&mut self) {
        let dyn_flags = self.dynamic_state_flags.clone();
        let has = |flag| dyn_flags.contains(flag);

        let default_viewport: vk::Viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .min_depth(0.0)
            .max_depth(1.0)
            .width(self.renderer.swapchain_extent.width as f32)
            .height(self.renderer.swapchain_extent.height as f32);

        if !has(DynamicStateFlags::VIEWPORTS) { self.set_viewports(&[default_viewport]); }
        if !has(DynamicStateFlags::SCISSORS) { self.set_scissors(&[self.renderer.swapchain_extent.into()]); }
        if !has(DynamicStateFlags::POLYGON_MODE) { self.set_polygon_mode(vk::PolygonMode::FILL); }
        if !has(DynamicStateFlags::PRIMITIVE_TOPOLOGY) { self.set_primitive_topology(vk::PrimitiveTopology::TRIANGLE_LIST); }
        if !has(DynamicStateFlags::PRIMITIVE_RESTART_ENABLE) { self.set_primitive_restart_enable(false); }
        if !has(DynamicStateFlags::DEPTH_TEST_ENABLE) { self.set_depth_test_enable(false); }
        if !has(DynamicStateFlags::DEPTH_WRITE_ENABLE) { self.set_depth_write_enable(false); }
        if !has(DynamicStateFlags::DEPTH_BIAS_ENABLE) { self.set_depth_bias_enable(false); }
        if !has(DynamicStateFlags::STENCIL_TEST_ENABLE) { self.set_stencil_test_enable(false); }
        if !has(DynamicStateFlags::RASTERIZER_DISCARD_ENABLE) { self.set_rasterizer_discard_enable(false); }
        if !has(DynamicStateFlags::RASTERIZATION_SAMPLES) { self.set_rasterization_samples(vk::SampleCountFlags::TYPE_1); }
        if !has(DynamicStateFlags::SAMPLE_MASK) { self.set_sample_mask(vk::SampleCountFlags::TYPE_1, &[vk::SampleMask::max_value()]); }
        if !has(DynamicStateFlags::ALPHA_TO_COVERAGE_ENABLE) { self.set_alpha_to_coverage_enable(false); }
        if !has(DynamicStateFlags::FRONT_FACE) { self.set_front_face(vk::FrontFace::COUNTER_CLOCKWISE); }
        if !has(DynamicStateFlags::CULL_MODE) { self.set_cull_mode(vk::CullModeFlags::NONE); }
        if !has(DynamicStateFlags::COLOR_BLEND_ENABLE) { self.set_color_blend_enable(&[0]); }
        if !has(DynamicStateFlags::COLOR_BLEND_EQUATION) { self.set_color_blend_equation(&[vk::ColorBlendEquationEXT::default()]); }
        if !has(DynamicStateFlags::COLOR_WRITE_MASK) { self.set_color_write_mask(&[vk::ColorComponentFlags::RGBA]); }
        // TODO: match this with the exact rules when things should be defined
    }

    pub fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        self.apply_unset_defaults();
        unsafe {
            self.renderer.device.cmd_draw(
                self.renderer.command_buffer,
                vertex_count,
                1,
                first_vertex,
                0,
            )
        };
    }

    pub fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        self.apply_unset_defaults();
        unsafe {
            self.renderer.device.cmd_draw_indexed(
                self.renderer.command_buffer,
                index_count,
                1,
                first_index,
                vertex_offset,
                0,
            )
        };
    }

    pub fn bind_vs_fs(&self, vs: vk::ShaderEXT, fs: vk::ShaderEXT) {
        let stages = [vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT];
        let shaders = [vs, fs];
        unsafe {
            self.renderer.device.shader_device.cmd_bind_shaders(
                self.renderer.command_buffer,
                &stages,
                &shaders,
            )
        };
    }

    pub fn set_vertex_input(&self, vertex_stride: u32, offsets: &[(u32, vk::Format)]) {
        let binding = [vk::VertexInputBindingDescription2EXT::default()
            .binding(0)
            .stride(vertex_stride)
            .input_rate(vk::VertexInputRate::VERTEX)
            .divisor(1)];
        let attribute: Vec<_> = offsets
            .into_iter()
            .enumerate()
            .map(|(i, (off, fmt))| {
                vk::VertexInputAttributeDescription2EXT::default()
                    .location(i as u32)
                    .binding(0)
                    .format(*fmt)
                    .offset(*off)
            })
            .collect();
        unsafe {
            self.renderer.device.shader_device.cmd_set_vertex_input(
                self.renderer.command_buffer,
                &binding,
                &attribute,
            )
        };
    }

    pub fn bind_index_buffer(&self, buffer: vk::Buffer, offset: u64) {
        unsafe {
            self.renderer.device.cmd_bind_index_buffer(
                self.renderer.command_buffer,
                buffer,
                offset,
                vk::IndexType::UINT16,
            )
        };
    }

    pub fn bind_vertex_buffer(&self, buffer: vk::Buffer) {
        let buffers = [buffer];
        let offsets = [0];
        unsafe {
            self.renderer.device.cmd_bind_vertex_buffers(
                self.renderer.command_buffer,
                0,
                &buffers,
                &offsets,
            )
        }; //, Some(&sizes), Some(&strides))};
    }

    pub fn bind_descriptor_set(
        &self,
        descriptor_set: vk::DescriptorSet,
        pipeline_layout: vk::PipelineLayout,
    ) {
        let descriptor_set = [descriptor_set];
        unsafe {
            self.renderer.device.cmd_bind_descriptor_sets(
                self.renderer.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &descriptor_set,
                &[],
            )
        };
    }

    pub fn push_constant<T>(&self, pipeline_layout: vk::PipelineLayout, data: &T) {
        let ptr = core::ptr::from_ref(data);
        let byte_ptr = unsafe { core::mem::transmute::<*const T, *const u8>(ptr) };
        let bytes = unsafe { core::slice::from_raw_parts(byte_ptr, size_of::<T>()) };
        unsafe {
            self.renderer.device.cmd_push_constants(
                self.renderer.command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes,
            )
        };
    }

    pub fn end_rendering(&self) {
        // end rendering
        unsafe {
            self.renderer
                .khr_dynamic_rendering
                .cmd_end_rendering(self.renderer.command_buffer)
        };
    }

    /// returns false if window redraw is required
    pub fn end_frame(self) -> bool {
        let renderer = &self.renderer;
        let swap_idx = self.swap_idx;

        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        // end frame
        let image_memory_barriers = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(renderer.swapchain_images[swap_idx as usize])
            .subresource_range(subrange)];
        unsafe {
            renderer.device.cmd_pipeline_barrier(
                renderer.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &image_memory_barriers,
            )
        };

        // end command buffer
        unsafe { renderer.device.end_command_buffer(renderer.command_buffer) }.unwrap();

        // submit queue
        let wait_semaphores = [renderer.ready_to_submit];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [renderer.command_buffer];
        let signal_semaphores = [renderer.ready_to_present];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        unsafe {
            renderer
                .device
                .queue_submit(renderer.queue, &[submit_info], renderer.ready_to_record)
        }
        .unwrap();

        let swapchains = [renderer.swapchain];
        let image_indices = [swap_idx];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        match unsafe {
            renderer
                .khr_swapchain
                .queue_present(renderer.queue, &present_info)
        } {
            Ok(false) => return true,
            Ok(true) => {
                println!("resize! (queue present suboptimal)");
                self.renderer.recreate_swapchain();
                return false;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                println!("resize! (queue present out of date)");
                self.renderer.recreate_swapchain();
                return false;
            }
            Err(e) => panic!("queue present error: {e}"),
        }
    }
}

bitflags::bitflags! {
    #[derive(Clone)]
    pub struct DynamicStateFlags: u32 {
        const VIEWPORTS                 = 1<< 0;
        const SCISSORS                  = 1<< 1;
        const POLYGON_MODE              = 1<< 2;
        const PRIMITIVE_TOPOLOGY        = 1<< 3;
        const PRIMITIVE_RESTART_ENABLE  = 1<< 4;
        const DEPTH_TEST_ENABLE         = 1<< 5;
        const DEPTH_WRITE_ENABLE        = 1<< 6;
        const DEPTH_BIAS_ENABLE         = 1<< 7;
        const STENCIL_TEST_ENABLE       = 1<< 8;
        const RASTERIZER_DISCARD_ENABLE = 1<< 9;
        const RASTERIZATION_SAMPLES     = 1<<10;
        const SAMPLE_MASK               = 1<<11;
        const ALPHA_TO_COVERAGE_ENABLE  = 1<<12;
        const FRONT_FACE                = 1<<13;
        const CULL_MODE                 = 1<<14;
        const COLOR_BLEND_ENABLE        = 1<<15;
        const COLOR_BLEND_EQUATION      = 1<<16;
        const COLOR_WRITE_MASK          = 1<<17;
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
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let (vs, fs) = (self.vs, self.fs);
                let mut frame = self.wait_and_begin_frame();

                frame.begin_rendering([
                    0.,
                    // (0x32 as f32 / 0xFF as f32).powf(2.2),
                    (0x30 as f32 / 0xFF as f32).powf(2.2),
                    (0x2f as f32 / 0xFF as f32).powf(2.2),
                    1.0,
                ]);
                frame.bind_vs_fs(vs, fs);

                frame.draw(3, 0);
                frame.end_rendering();
                if !frame.end_frame() {}

                // self.window.pre_present_notify();
                self.window.request_redraw();
            }
            _ => {}
        }
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
