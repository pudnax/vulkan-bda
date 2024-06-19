use std::{collections::VecDeque, slice, sync::Arc};

use ash::{
    khr::{self},
    prelude::VkResult,
    vk,
};

use crate::{
    device::{Device, DeviceExt},
    surface::Surface,
};

pub struct Frame {
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    present_finished: vk::Fence,
    device: Arc<ash::Device>,
}

impl Frame {
    fn destroy(&mut self, pool: &vk::CommandPool) {
        unsafe {
            self.device.destroy_fence(self.present_finished, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device
                .free_command_buffers(*pool, &[self.command_buffer]);
        }
    }
}

impl Frame {
    fn new(device: &Arc<ash::Device>, command_pool: &vk::CommandPool) -> VkResult<Self> {
        let present_finished = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::default()),
                None,
            )
        }?;
        let image_available_semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let render_finished_semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }?[0];
        Ok(Self {
            command_buffer,
            image_available_semaphore,
            render_finished_semaphore,
            present_finished,
            device: device.clone(),
        })
    }
}

pub struct FrameGuard {
    frame: Frame,
    dyn_state: DynamicStateFlags,
    extent: vk::Extent2D,
    image_idx: usize,
    device: Arc<ash::Device>,
    ext: Arc<DeviceExt>,
}

pub struct Swapchain {
    images: Vec<vk::Image>,
    views: Vec<vk::ImageView>,
    frames: VecDeque<Frame>,
    command_pool: vk::CommandPool,
    current_image: usize,
    format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    inner: vk::SwapchainKHR,
    loader: khr::swapchain::Device,
    device: Arc<ash::Device>,
    ext: Arc<DeviceExt>,
}

impl Swapchain {
    const SUBRANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    pub fn new(
        device: &Device,
        surface: &Surface,
        swapchain_loader: khr::swapchain::Device,
    ) -> VkResult<Self> {
        let info = surface.info(device);
        let capabilities = info.capabilities;
        let format = info
            .formats
            .iter()
            .find(|format| {
                matches!(
                    format.format,
                    vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
                )
            })
            .unwrap_or(&info.formats[0]);
        let image_count = 3
            .max(capabilities.min_image_count)
            .min(capabilities.max_image_count);

        let queue_family_index = [device.queue_family];
        let swapchain_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let extent = capabilities.current_extent;
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(**surface)
            .image_format(format.format)
            .image_usage(swapchain_usage)
            .image_extent(extent)
            .image_color_space(format.color_space)
            .min_image_count(image_count)
            .image_array_layers(capabilities.max_image_array_layers)
            .queue_family_indices(&queue_family_index)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(capabilities.supported_composite_alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let views = images
            .iter()
            .map(|img| {
                let info = vk::ImageViewCreateInfo::default()
                    .image(*img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { device.create_image_view(&info, None) }
            })
            .collect::<VkResult<Vec<_>>>()?;

        let frames = VecDeque::new();

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(device.queue_family),
                None,
            )?
        };

        Ok(Self {
            images,
            views,
            frames,
            command_pool,
            current_image: 0,
            format: *format,
            extent,
            inner: swapchain,
            loader: swapchain_loader,
            device: device.device.clone(),
            ext: device.ext.clone(),
        })
    }

    pub fn destroy(&self) {
        for view in self.views.iter() {
            unsafe { self.device.destroy_image_view(*view, None) };
        }
        unsafe { self.loader.destroy_swapchain(self.inner, None) };
    }

    pub fn recreate(&mut self, device: &Device, surface: &Surface) -> VkResult<()> {
        let info = surface.info(device);
        let capabilities = info.capabilities;

        for view in self.views.iter() {
            unsafe { self.device.destroy_image_view(*view, None) };
        }
        let old_swapchain = self.inner;

        let queue_family_index = [device.queue_family];
        let swapchain_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        self.extent = capabilities.current_extent;
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(**surface)
            .old_swapchain(old_swapchain)
            .image_format(self.format.format)
            .image_usage(swapchain_usage)
            .image_extent(self.extent)
            .image_color_space(self.format.color_space)
            .min_image_count(self.images.len() as u32)
            .image_array_layers(capabilities.max_image_array_layers)
            .queue_family_indices(&queue_family_index)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(capabilities.supported_composite_alpha)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        self.inner = unsafe { self.loader.create_swapchain(&swapchain_create_info, None)? };

        unsafe { self.loader.destroy_swapchain(old_swapchain, None) };

        self.images = unsafe { self.loader.get_swapchain_images(self.inner)? };
        self.views = self
            .images
            .iter()
            .map(|img| {
                let info = vk::ImageViewCreateInfo::default()
                    .image(*img)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { device.create_image_view(&info, None) }
            })
            .collect::<VkResult<Vec<_>>>()?;
        Ok(())
    }

    pub fn get_current_frame(&self) -> Option<&Frame> {
        self.frames.back()
    }
    pub fn get_current_image(&self) -> &vk::Image {
        &self.images[self.current_image]
    }
    pub fn get_current_image_view(&self) -> &vk::ImageView {
        &self.views[self.current_image]
    }

    pub fn acquire_next_image(&mut self) -> VkResult<FrameGuard> {
        self.frames.retain_mut(|frame| {
            let status = unsafe { self.device.get_fence_status(frame.present_finished) };
            if status == Ok(true) {
                frame.destroy(&self.command_pool);
                false
            } else {
                true
            }
        });

        let mut frame = Frame::new(&self.device, &self.command_pool)?;

        let idx = match unsafe {
            self.loader.acquire_next_image(
                self.inner,
                u64::MAX,
                frame.image_available_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((idx, false)) => idx,
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                frame.destroy(&self.command_pool);
                return VkResult::Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
            }
            Err(e) => return Err(e),
        };

        self.current_image = idx as usize;
        unsafe {
            self.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default(),
            )?
        };

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .subresource_range(Self::SUBRANGE)
            .image(self.images[self.current_image])
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_barrier));
        unsafe {
            self.device
                .cmd_pipeline_barrier2(frame.command_buffer, &dependency_info)
        };

        Ok(FrameGuard {
            frame,
            dyn_state: DynamicStateFlags::empty(),
            extent: self.extent,
            image_idx: self.current_image,
            device: self.device.clone(),
            ext: self.ext.clone(),
        })
    }

    pub fn submit_image(&mut self, queue: &vk::Queue, frame_guard: FrameGuard) -> VkResult<()> {
        let frame = frame_guard.frame;

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .subresource_range(Self::SUBRANGE)
            .image(self.images[frame_guard.image_idx])
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_barrier));
        unsafe {
            self.device
                .cmd_pipeline_barrier2(frame.command_buffer, &dependency_info)
        };

        unsafe { self.device.end_command_buffer(frame.command_buffer) }?;

        let wait_semaphores = [frame.image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame.render_finished_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(slice::from_ref(&frame.command_buffer))
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device
                .queue_submit(*queue, &[submit_info], frame.present_finished)?
        };

        self.frames.push_back(frame);
        let image_indices = [frame_guard.image_idx as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(slice::from_ref(&self.inner))
            .image_indices(&image_indices);
        match unsafe { self.loader.queue_present(*queue, &present_info) } {
            Ok(false) => Ok(()),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                VkResult::Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
            }
            Err(e) => Err(e),
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for view in self.views.iter() {
                self.device.destroy_image_view(*view, None);
            }
            self.loader.destroy_swapchain(self.inner, None);
            self.frames
                .iter_mut()
                .for_each(|f| f.destroy(&self.command_pool));
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl FrameGuard {
    pub fn begin_rendering(&mut self, view: &vk::ImageView, color: [f32; 4]) {
        self.dyn_state = DynamicStateFlags::empty();

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue { float32: color },
        };
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(*view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_image_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.extent.into())
            .layer_count(1)
            .color_attachments(&color_attachments);
        unsafe {
            self.ext
                .dynamic_rendering
                .cmd_begin_rendering(self.frame.command_buffer, &rendering_info)
        };
    }

    pub fn set_viewports(&mut self, viewports: &[vk::Viewport]) {
        self.dyn_state |= DynamicStateFlags::VIEWPORTS;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_viewport_with_count(self.frame.command_buffer, viewports)
        };
    }
    pub fn set_scissors(&mut self, scissors: &[vk::Rect2D]) {
        self.dyn_state |= DynamicStateFlags::SCISSORS;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_scissor_with_count(self.frame.command_buffer, scissors)
        };
    }
    pub fn set_polygon_mode(&mut self, mode: vk::PolygonMode) {
        self.dyn_state |= DynamicStateFlags::POLYGON_MODE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_polygon_mode(self.frame.command_buffer, mode)
        };
    }
    pub fn set_primitive_topology(&mut self, topology: vk::PrimitiveTopology) {
        self.dyn_state |= DynamicStateFlags::PRIMITIVE_TOPOLOGY;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_primitive_topology(self.frame.command_buffer, topology)
        };
    }
    pub fn set_primitive_restart_enable(&mut self, enabled: bool) {
        self.dyn_state |= DynamicStateFlags::PRIMITIVE_RESTART_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_primitive_restart_enable(self.frame.command_buffer, enabled)
        };
    }
    pub fn set_depth_test_enable(&mut self, enabled: bool) {
        self.dyn_state |= DynamicStateFlags::DEPTH_TEST_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_depth_test_enable(self.frame.command_buffer, enabled)
        };
    }
    pub fn set_depth_write_enable(&mut self, enabled: bool) {
        self.dyn_state |= DynamicStateFlags::DEPTH_WRITE_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_depth_write_enable(self.frame.command_buffer, enabled)
        };
    }
    pub fn set_depth_bias_enable(&mut self, enabled: bool) {
        self.dyn_state |= DynamicStateFlags::DEPTH_BIAS_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_depth_bias_enable(self.frame.command_buffer, enabled)
        };
    }
    pub fn set_stencil_test_enable(&mut self, enabled: bool) {
        self.dyn_state |= DynamicStateFlags::STENCIL_TEST_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_stencil_test_enable(self.frame.command_buffer, enabled)
        };
    }
    pub fn set_rasterizer_discard_enable(&mut self, enabled: bool) {
        self.dyn_state |= DynamicStateFlags::RASTERIZER_DISCARD_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_rasterizer_discard_enable(self.frame.command_buffer, enabled)
        };
    }
    pub fn set_rasterization_samples(&mut self, sample_count_flags: vk::SampleCountFlags) {
        self.dyn_state |= DynamicStateFlags::RASTERIZATION_SAMPLES;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_rasterization_samples(self.frame.command_buffer, sample_count_flags)
        };
    }
    pub fn set_sample_mask(
        &mut self,
        samples: vk::SampleCountFlags,
        sample_mask: &[vk::SampleMask],
    ) {
        self.dyn_state |= DynamicStateFlags::SAMPLE_MASK;
        unsafe {
            self.ext.shader_object.cmd_set_sample_mask(
                self.frame.command_buffer,
                samples,
                sample_mask,
            )
        };
    }
    pub fn set_alpha_to_coverage_enable(&mut self, enable: bool) {
        self.dyn_state |= DynamicStateFlags::ALPHA_TO_COVERAGE_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_alpha_to_coverage_enable(self.frame.command_buffer, enable)
        };
    }
    pub fn set_front_face(&mut self, front_face: vk::FrontFace) {
        self.dyn_state |= DynamicStateFlags::FRONT_FACE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_front_face(self.frame.command_buffer, front_face);
        }
    }
    pub fn set_cull_mode(&mut self, cullmode: vk::CullModeFlags) {
        self.dyn_state |= DynamicStateFlags::CULL_MODE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_cull_mode(self.frame.command_buffer, cullmode)
        };
    }
    pub fn set_color_blend_enable(&mut self, enables: &[u32]) {
        self.dyn_state |= DynamicStateFlags::COLOR_BLEND_ENABLE;
        unsafe {
            self.ext
                .shader_object
                .cmd_set_color_blend_enable(self.frame.command_buffer, 0, enables)
        };
    }
    pub fn set_color_blend_equation(&mut self, equations: &[vk::ColorBlendEquationEXT]) {
        self.dyn_state |= DynamicStateFlags::COLOR_BLEND_EQUATION;
        unsafe {
            self.ext.shader_object.cmd_set_color_blend_equation(
                self.frame.command_buffer,
                0,
                equations,
            )
        };
    }
    pub fn set_color_write_mask(&mut self, write_masks: &[vk::ColorComponentFlags]) {
        self.dyn_state |= DynamicStateFlags::COLOR_WRITE_MASK;
        unsafe {
            self.ext.shader_object.cmd_set_color_write_mask(
                self.frame.command_buffer,
                0,
                write_masks,
            )
        };
    }

    #[rustfmt::skip]
    fn apply_unset_defaults(&mut self) {
        let dyn_flags = self.dyn_state.clone();
        let has = |flag| dyn_flags.contains(flag);

        let default_viewport: vk::Viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .min_depth(0.0)
            .max_depth(1.0)
            .width(self.extent.width as f32)
            .height(self.extent.height as f32);

        if !has(DynamicStateFlags::VIEWPORTS) { self.set_viewports(&[default_viewport]); }
        if !has(DynamicStateFlags::SCISSORS) { self.set_scissors(&[self.extent.into()]); }
        if !has(DynamicStateFlags::POLYGON_MODE) { self.set_polygon_mode(vk::PolygonMode::FILL); }
        if !has(DynamicStateFlags::PRIMITIVE_TOPOLOGY) { self.set_primitive_topology(vk::PrimitiveTopology::TRIANGLE_LIST); }
        if !has(DynamicStateFlags::PRIMITIVE_RESTART_ENABLE) { self.set_primitive_restart_enable(false); }
        if !has(DynamicStateFlags::DEPTH_TEST_ENABLE) { self.set_depth_test_enable(false); }
        if !has(DynamicStateFlags::DEPTH_WRITE_ENABLE) { self.set_depth_write_enable(false); }
        if !has(DynamicStateFlags::DEPTH_BIAS_ENABLE) { self.set_depth_bias_enable(false); }
        if !has(DynamicStateFlags::STENCIL_TEST_ENABLE) { self.set_stencil_test_enable(false); }
        if !has(DynamicStateFlags::RASTERIZER_DISCARD_ENABLE) { self.set_rasterizer_discard_enable(false); }
        if !has(DynamicStateFlags::RASTERIZATION_SAMPLES) { self.set_rasterization_samples(vk::SampleCountFlags::TYPE_1); }
        if !has(DynamicStateFlags::SAMPLE_MASK) { self.set_sample_mask(vk::SampleCountFlags::TYPE_1, &[vk::SampleMask::MAX]); }
        if !has(DynamicStateFlags::ALPHA_TO_COVERAGE_ENABLE) { self.set_alpha_to_coverage_enable(false); }
        if !has(DynamicStateFlags::FRONT_FACE) { self.set_front_face(vk::FrontFace::COUNTER_CLOCKWISE); }
        if !has(DynamicStateFlags::CULL_MODE) { self.set_cull_mode(vk::CullModeFlags::NONE); }
        if !has(DynamicStateFlags::COLOR_BLEND_ENABLE) { self.set_color_blend_enable(&[0]); }
        if !has(DynamicStateFlags::COLOR_BLEND_EQUATION) { self.set_color_blend_equation(&[vk::ColorBlendEquationEXT::default()]); }
        if !has(DynamicStateFlags::COLOR_WRITE_MASK) { self.set_color_write_mask(&[vk::ColorComponentFlags::RGBA]); }
    }

    pub fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        self.apply_unset_defaults();
        unsafe {
            self.device
                .cmd_draw(self.frame.command_buffer, vertex_count, 1, first_vertex, 0)
        };
    }

    pub fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        self.apply_unset_defaults();
        unsafe {
            self.device.cmd_draw_indexed(
                self.frame.command_buffer,
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
            self.ext
                .shader_object
                .cmd_bind_shaders(self.frame.command_buffer, &stages, &shaders)
        };
    }

    pub fn set_vertex_input(&self, vertex_stride: u32, offsets: &[(u32, vk::Format)]) {
        let binding = [vk::VertexInputBindingDescription2EXT::default()
            .binding(0)
            .stride(vertex_stride)
            .input_rate(vk::VertexInputRate::VERTEX)
            .divisor(1)];
        let attribute: Vec<_> = offsets
            .iter()
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
            self.ext.shader_object.cmd_set_vertex_input(
                self.frame.command_buffer,
                &binding,
                &attribute,
            )
        };
    }

    pub fn bind_index_buffer(&self, buffer: vk::Buffer, offset: u64) {
        unsafe {
            self.device.cmd_bind_index_buffer(
                self.frame.command_buffer,
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
            self.device
                .cmd_bind_vertex_buffers(self.frame.command_buffer, 0, &buffers, &offsets)
        };
    }

    pub fn bind_descriptor_set(
        &self,
        bind_point: vk::PipelineBindPoint,
        descriptor_set: vk::DescriptorSet,
        pipeline_layout: vk::PipelineLayout,
    ) {
        let descriptor_set = [descriptor_set];
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                self.frame.command_buffer,
                bind_point,
                pipeline_layout,
                0,
                &descriptor_set,
                &[],
            )
        };
    }

    pub fn push_constant<T>(
        &self,
        pipeline_layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        data: &[T],
    ) {
        let ptr = core::ptr::from_ref(data);
        let bytes = unsafe { core::slice::from_raw_parts(ptr.cast(), size_of::<T>() * data.len()) };
        unsafe {
            self.device.cmd_push_constants(
                self.frame.command_buffer,
                pipeline_layout,
                stages,
                0,
                bytes,
            )
        };
    }

    pub fn end_rendering(&mut self) {
        unsafe {
            self.ext
                .dynamic_rendering
                .cmd_end_rendering(self.frame.command_buffer)
        };
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
