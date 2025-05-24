from .md import *
CATEGORY_TYPE = "🎈LAOGOU/Utils"

pb_id_cnt = time.time()
preview_bridge_image_id_map = {}
preview_bridge_image_name_map = {}
preview_bridge_cache = {}
preview_bridge_last_mask_cache = {}

def set_previewbridge_image(node_id, file, item):
    global pb_id_cnt

    if file in preview_bridge_image_name_map:
        pb_id = preview_bridge_image_name_map[node_id, file]
        if pb_id.startswith(f"${node_id}"):
            return pb_id

    pb_id = f"${node_id}-{pb_id_cnt}"
    
    preview_bridge_image_id_map[pb_id] = (file, item)
    preview_bridge_image_name_map[node_id, file] = (pb_id, item)
    
    pb_id_cnt += 1

    return pb_id


class CachePreviewBridge:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("STRING", {"default": ""}),
                    "use_cache": ("BOOLEAN", {"default": True, "label_on": "Cache", "label_off": "Input"}),
                    },
                "optional": {
                    "images": ("IMAGE",),
                    },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "doit"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_TYPE

    def __init__(self):
        super().__init__()
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prev_hash = None
    @staticmethod
    def load_image(pb_id):
        clipspace_dir = os.path.join(folder_paths.get_input_directory(), "clipspace")
        files = [f for f in os.listdir(clipspace_dir) 
                if f.lower().startswith('clipspace') and f.lower().endswith('.png')]
        
        # 初始化默认值
        image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
        mask = torch.zeros((512, 512), dtype=torch.float32, device="cpu")
        ui_item = {
            "filename": 'empty.png',
            "subfolder": '',
            "type": 'temp'
        }
        
        if files:
            latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(clipspace_dir, f)))
            latest_path = os.path.join(clipspace_dir, latest_file)
            latest_mtime = os.path.getmtime(latest_path)
            
            current_path = None
            if pb_id in preview_bridge_image_id_map:
                current_path, ui_item = preview_bridge_image_id_map[pb_id]
                current_mtime = os.path.getmtime(current_path) if os.path.exists(current_path) else 0
            
            if current_path is None or latest_mtime > current_mtime:
                preview_dir = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge')
                os.makedirs(preview_dir, exist_ok=True)
                
                new_filename = f"PB-{os.path.splitext(latest_file)[0]}.png"
                new_path = os.path.join(preview_dir, new_filename)
                
                # 检查是否存在同名文件
                if os.path.exists(new_path):
                    if pb_id in preview_bridge_image_id_map:
                        current_path, ui_item = preview_bridge_image_id_map[pb_id]
                        if os.path.exists(current_path):
                            new_path = current_path
                        else:
                            import shutil
                            shutil.copy2(latest_path, new_path)
                            ui_item = {
                                "filename": new_filename,
                                "subfolder": 'PreviewBridge',
                                "type": 'temp'
                            }
                            preview_bridge_image_id_map[pb_id] = (new_path, ui_item)
                else:
                    import shutil
                    shutil.copy2(latest_path, new_path)
                    ui_item = {
                        "filename": new_filename,
                        "subfolder": 'PreviewBridge',
                        "type": 'temp'
                    }
                    preview_bridge_image_id_map[pb_id] = (new_path, ui_item)
                
                current_path = new_path
            
            try:
                i = Image.open(current_path)
                i = ImageOps.exif_transpose(i)
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]

                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((image.shape[1], image.shape[2]), dtype=torch.float32, device="cpu")
            except Exception as e:
                print(f"Error loading image: {e}")
                # 出错时使用默认值

        final_mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
        return image, final_mask, ui_item

    def doit(self, image, use_cache, unique_id, images=None, extra_pnginfo=None):
        # 检查是否有新的剪贴板图片需要更新缓存
        if images is None and image and image not in preview_bridge_image_id_map:
            print("Component value changed, updating cache")
            # 生成新的pb_id并更新缓存
            pb_id = f"${unique_id}-{time.time()}"
            pixels, mask, path_item = CachePreviewBridge.load_image(pb_id)
            if path_item["type"] != "temp" or path_item["filename"] != "empty.png":
                return {
                    "ui": {"images": [path_item]},
                    "result": (pixels, mask),
                }

        # 原有的判断逻辑
        if images is None and not image:
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
            empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device="cpu")
            return {
                "ui": {"images": []},
                "result": (empty_image, empty_mask),
            }

        if use_cache:
            if not image:
                
                empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
                empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device="cpu")
                return {
                    "ui": {"images": []},
                    "result": (empty_image, empty_mask),
                }

            if image.startswith('$'):
                node_id = image.split('-')[0][1:]
                related_pb_ids = [pb_id for pb_id in preview_bridge_image_id_map.keys() 
                                if pb_id.startswith(f"${node_id}-")]
                
                if related_pb_ids:
                    latest_pb_id = max(related_pb_ids, key=lambda x: float(x.split('-')[1]))
                    image = latest_pb_id
                    pixels, mask, path_item = CachePreviewBridge.load_image(image)
                    return {
                        "ui": {"images": [path_item]},
                        "result": (pixels, mask),
                    }

        # 非缓存模式或需要创建新缓存
        if images is not None:
            mask = torch.zeros((1, images.shape[1], images.shape[2]), dtype=torch.float32, device="cpu")
            res = PreviewImage().save_images(
                images, 
                filename_prefix=f"PreviewBridge/PB-{unique_id}-", 
                extra_pnginfo=extra_pnginfo
            )

            image2 = res['ui']['images']
            pixels = images

            path = os.path.join(folder_paths.get_temp_directory(), 'PreviewBridge', image2[0]['filename'])
            pb_id = set_previewbridge_image(unique_id, path, image2[0])
            
            preview_bridge_cache[unique_id] = (images, image2)
            
            return {
                "ui": {"images": image2},
                "result": (pixels, mask),
            }
        
        # 如果走到这里,说明既没有有效的缓存也没有新的图像输入
        empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
        empty_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device="cpu")
        return {
            "ui": {"images": []},
            "result": (empty_image, empty_mask),
        }
    
class LG_Noise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["fade", "dissolve", "gaussian"], ),
                "opacity": ("FLOAT", { "default": 1.0, "min": 0, "max": 1, "step": 0.01 }),
                "strength": ("INT", { "default": 1, "min": 1, "max": 32, "step": 1 }),
                "density": ("FLOAT", { "default": 1.0, "min": 0, "max": 1, "step": 0.05 }),
                "sharpen": ("INT", { "default": 0, "min": -32, "max": 32, "step": 1 }),
                "brightness": ("FLOAT", { "default": 1.0, "min": 0, "max": 3, "step": 0.05 }),
                "random_color": ("BOOLEAN", {"default": True}),
                "color": ("COLOR", {"default": "#808080"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7fffffff}),
            },
            "optional": {
                "image_optional": ("IMAGE",),
                "mask_optional": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_noise"
    CATEGORY = CATEGORY_TYPE
    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return torch.tensor([r, g, b])
    def make_noise(self, type, opacity, density, strength, sharpen, random_color, color, brightness, seed,
                image_optional=None, mask_optional=None):
        if image_optional is None:
            image = torch.zeros([1, 1, 1, 3])
        else:
            image = image_optional
        h, w = image.shape[1:3]
        if h == 1 and w == 1:
            image = image.repeat(1, 512, 512, 1)
            h, w = 512, 512
        if mask_optional is not None:
            mask = mask_optional.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            mask = torch.ones((1, h, w, 1), device=image.device).repeat(1, 1, 1, 3)
        if seed == -1:
            seed = torch.randint(0, 0x7fffffff, (1,)).item()
        torch.manual_seed(seed)
        print(f"[LG_Noise] Using seed: {seed}")
        color_rgb = self.hex_to_rgb(color).to(image.device)
        def generate_noise(size_h, size_w, is_gaussian=False):
            """统一的噪声生成函数"""
            if random_color:
                if is_gaussian:
                    noise = torch.randn(1, size_h, size_w, 3, device=image.device)
                    return torch.clamp(noise * 0.5 + 0.5, 0, 1)
                else:
                    return torch.rand(1, size_h, size_w, 3, device=image.device)
            else:
                return torch.ones(1, size_h, size_w, 3, device=image.device)
        def generate_density_mask(size_h, size_w):
            """统一的密度遮罩生成函数"""
            return (torch.rand(1, size_h, size_w, 1, device=image.device) < density).float()
        if strength > 1:
            small_h, small_w = h // strength, w // strength
            density_mask = generate_density_mask(small_h, small_w)
            noise = generate_noise(small_h, small_w) * density_mask
            noise = torch.nn.functional.interpolate(
                noise.permute(0, 3, 1, 2),
                size=(h, w),
                mode='nearest'
            ).permute(0, 2, 3, 1)
            if type != "fade":
                density_mask = torch.nn.functional.interpolate(
                    density_mask.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='nearest'
                ).permute(0, 2, 3, 1)
        else:
            density_mask = generate_density_mask(h, w)
            noise = generate_noise(h, w) * density_mask
        colored_noise = noise * color_rgb.view(1, 1, 1, 3) * brightness
        if type == "fade":
            result = image * (1 - opacity) + colored_noise * opacity
        elif type == "dissolve":
            dissolve_mask = (torch.rand(1, h//strength if strength > 1 else h,
                                    w//strength if strength > 1 else w, 1,
                                    device=image.device) < opacity).float()
            density_mask = (torch.rand(1, h//strength if strength > 1 else h,
                                    w//strength if strength > 1 else w, 1,
                                    device=image.device) < density).float()
            dissolve_mask = dissolve_mask * density_mask
            if strength > 1:
                dissolve_mask = torch.nn.functional.interpolate(
                    dissolve_mask.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='nearest'
                ).permute(0, 2, 3, 1)
            dissolve_mask = dissolve_mask.repeat(1, 1, 1, 3)
            if random_color:
                noise = torch.rand(1, h, w, 3, device=image.device)
            else:
                noise = torch.ones(1, h, w, 3, device=image.device)
            noise = noise * dissolve_mask
            colored_noise = noise * color_rgb.view(1, 1, 1, 3) * brightness
            result = image * (1-dissolve_mask) + colored_noise * dissolve_mask
        elif type == "gaussian":
            noise = generate_noise(h//strength if strength > 1 else h,
                                w//strength if strength > 1 else w,
                                is_gaussian=True)
            if strength > 1:
                noise = torch.nn.functional.interpolate(
                    noise.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='nearest'
                ).permute(0, 2, 3, 1)
            noise = (noise - 0.5) * opacity * 2
            noise = noise * density_mask
            colored_noise = noise * color_rgb.view(1, 1, 1, 3) * brightness
            result = torch.clamp(image + colored_noise, 0, 1)
        result = torch.clamp(result, 0, 1)
        if sharpen != 0:
            kernel_size = abs(sharpen) * 2 + 1
            noise_part = result * mask - image * mask
            if sharpen < 0:
                blurred_noise = T.functional.gaussian_blur(
                    noise_part.permute([0,3,1,2]),
                    kernel_size
                ).permute([0,2,3,1])
                result = image + blurred_noise
            else:
                blurred = T.functional.gaussian_blur(
                    noise_part.permute([0,3,1,2]),
                    kernel_size
                ).permute([0,2,3,1])
                sharpened_noise = noise_part + (noise_part - blurred) * (sharpen / 8)
                result = image + torch.clamp(sharpened_noise, 0, 1)
        result = torch.clamp(result, 0, 1)
        result = image * (1 - mask) + result * mask
        return (result,)
    
class LG_Noise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["fade", "dissolve", "gaussian"], ),
                "opacity": ("FLOAT", { "default": 1.0, "min": 0, "max": 1, "step": 0.01 }),
                "strength": ("INT", { "default": 1, "min": 1, "max": 32, "step": 1 }),
                "density": ("FLOAT", { "default": 1.0, "min": 0, "max": 1, "step": 0.05 }),
                "sharpen": ("INT", { "default": 0, "min": -32, "max": 32, "step": 1 }),
                "brightness": ("FLOAT", { "default": 1.0, "min": 0, "max": 3, "step": 0.05 }),
                "random_color": ("BOOLEAN", {"default": True}),
                "color": ("COLOR", {"default": "#808080"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0x7fffffff}),
            },
            "optional": {
                "image_optional": ("IMAGE",),
                "mask_optional": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_noise"
    CATEGORY = CATEGORY_TYPE
    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return torch.tensor([r, g, b])
    def make_noise(self, type, opacity, density, strength, sharpen, random_color, color, brightness, seed,
                image_optional=None, mask_optional=None):
        if image_optional is None:
            image = torch.zeros([1, 1, 1, 3])
        else:
            image = image_optional
        h, w = image.shape[1:3]
        if h == 1 and w == 1:
            image = image.repeat(1, 512, 512, 1)
            h, w = 512, 512
        if mask_optional is not None:
            mask = mask_optional.unsqueeze(-1).repeat(1, 1, 1, 3)
        else:
            mask = torch.ones((1, h, w, 1), device=image.device).repeat(1, 1, 1, 3)
        if seed == -1:
            seed = torch.randint(0, 0x7fffffff, (1,)).item()
        torch.manual_seed(seed)
        print(f"[LG_Noise] Using seed: {seed}")
        color_rgb = self.hex_to_rgb(color).to(image.device)
        def generate_noise(size_h, size_w, is_gaussian=False):
            """统一的噪声生成函数"""
            if random_color:
                if is_gaussian:
                    noise = torch.randn(1, size_h, size_w, 3, device=image.device)
                    return torch.clamp(noise * 0.5 + 0.5, 0, 1)
                else:
                    return torch.rand(1, size_h, size_w, 3, device=image.device)
            else:
                return torch.ones(1, size_h, size_w, 3, device=image.device)
        def generate_density_mask(size_h, size_w):
            """统一的密度遮罩生成函数"""
            return (torch.rand(1, size_h, size_w, 1, device=image.device) < density).float()
        if strength > 1:
            small_h, small_w = h // strength, w // strength
            density_mask = generate_density_mask(small_h, small_w)
            noise = generate_noise(small_h, small_w) * density_mask
            noise = torch.nn.functional.interpolate(
                noise.permute(0, 3, 1, 2),
                size=(h, w),
                mode='nearest'
            ).permute(0, 2, 3, 1)
            if type != "fade":
                density_mask = torch.nn.functional.interpolate(
                    density_mask.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='nearest'
                ).permute(0, 2, 3, 1)
        else:
            density_mask = generate_density_mask(h, w)
            noise = generate_noise(h, w) * density_mask
        colored_noise = noise * color_rgb.view(1, 1, 1, 3) * brightness
        if type == "fade":
            result = image * (1 - opacity) + colored_noise * opacity
        elif type == "dissolve":
            dissolve_mask = (torch.rand(1, h//strength if strength > 1 else h,
                                    w//strength if strength > 1 else w, 1,
                                    device=image.device) < opacity).float()
            density_mask = (torch.rand(1, h//strength if strength > 1 else h,
                                    w//strength if strength > 1 else w, 1,
                                    device=image.device) < density).float()
            dissolve_mask = dissolve_mask * density_mask
            if strength > 1:
                dissolve_mask = torch.nn.functional.interpolate(
                    dissolve_mask.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='nearest'
                ).permute(0, 2, 3, 1)
            dissolve_mask = dissolve_mask.repeat(1, 1, 1, 3)
            if random_color:
                noise = torch.rand(1, h, w, 3, device=image.device)
            else:
                noise = torch.ones(1, h, w, 3, device=image.device)
            noise = noise * dissolve_mask
            colored_noise = noise * color_rgb.view(1, 1, 1, 3) * brightness
            result = image * (1-dissolve_mask) + colored_noise * dissolve_mask
        elif type == "gaussian":
            noise = generate_noise(h//strength if strength > 1 else h,
                                w//strength if strength > 1 else w,
                                is_gaussian=True)
            if strength > 1:
                noise = torch.nn.functional.interpolate(
                    noise.permute(0, 3, 1, 2),
                    size=(h, w),
                    mode='nearest'
                ).permute(0, 2, 3, 1)
            noise = (noise - 0.5) * opacity * 2
            noise = noise * density_mask
            colored_noise = noise * color_rgb.view(1, 1, 1, 3) * brightness
            result = torch.clamp(image + colored_noise, 0, 1)
        result = torch.clamp(result, 0, 1)
        if sharpen != 0:
            kernel_size = abs(sharpen) * 2 + 1
            noise_part = result * mask - image * mask
            if sharpen < 0:
                blurred_noise = T.functional.gaussian_blur(
                    noise_part.permute([0,3,1,2]),
                    kernel_size
                ).permute([0,2,3,1])
                result = image + blurred_noise
            else:
                blurred = T.functional.gaussian_blur(
                    noise_part.permute([0,3,1,2]),
                    kernel_size
                ).permute([0,2,3,1])
                sharpened_noise = noise_part + (noise_part - blurred) * (sharpen / 8)
                result = image + torch.clamp(sharpened_noise, 0, 1)
        result = torch.clamp(result, 0, 1)
        result = image * (1 - mask) + result * mask
        return (result,)



WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']

class IPAdapterWeightTypes:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "weight_type": (WEIGHT_TYPES, ),
        }}
    
    RETURN_TYPES = (AlwaysEqualProxy('*'),)
    RETURN_NAMES = ("weight_type",)
    FUNCTION = "get_weight_types"
    CATEGORY = CATEGORY_TYPE

    def get_weight_types(self, weight_type):
        return (weight_type,)

NODE_CLASS_MAPPINGS = {
    "CachePreviewBridge": CachePreviewBridge,
    "LG_Noise": LG_Noise,
    "IPAdapterWeightTypes": IPAdapterWeightTypes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CachePreviewBridge": "🎈LG_PreviewBridge",
    "LG_Noise": "🎈LG_Noise",
    "IPAdapterWeightTypes": "🎈IPAdapter权重类型"
}

