from .md import *
CATEGORY_TYPE = "🎈LAOGOU/Utils"

def execute_command_with_realtime_output(cmd, cwd, message_path, clear_first=True, start_message=""):
    """通用的实时命令执行方法"""
    import subprocess
    import sys
    from server import PromptServer
    import comfy.model_management as model_management
    
    try:
        # 发送开始消息
        if start_message:
            PromptServer.instance.send_sync(message_path, {
                "text": start_message,
                "clear": clear_first
            })
        
        # 执行命令，实时输出
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时读取输出
        while True:
            # 检查是否被中断
            if model_management.processing_interrupted():
                process.terminate()
                PromptServer.instance.send_sync(message_path, {
                    "text": "🛑 用户中断了操作",
                    "clear": False
                })
                return False, "🛑 用户中断了操作"
            
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                line = output.strip()
                if line:
                    PromptServer.instance.send_sync(message_path, {
                        "text": line,
                        "clear": False
                    })
        
        # 检查返回码
        return_code = process.poll()
        return return_code == 0, return_code
        
    except Exception as e:
        error_msg = f"💥 执行异常: {str(e)}"
        PromptServer.instance.send_sync(message_path, {
            "text": error_msg,
            "clear": False
        })
        return False, str(e)

pb_id_cnt = time.time()
preview_bridge_image_id_map = {}
preview_bridge_image_name_map = {}
preview_bridge_cache = {}
preview_bridge_last_mask_cache = {}
# 存储每个节点的图片hash，用于检测图片内容变化
preview_bridge_image_hashes = {}

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
    def calculate_image_hash(images):
        """计算图片的哈希值用于检测是否改变"""
        try:
            if images is None:
                return None
            np_images = images.cpu().numpy()
            image_bytes = np_images.tobytes()
            return hashlib.md5(image_bytes).hexdigest()
        except:
            return None
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

    def doit(self, image, unique_id, images=None, extra_pnginfo=None):
        # 智能决定使用模式：基于图片内容hash检测
        current_hash = self.calculate_image_hash(images) if images is not None else None
        last_hash = preview_bridge_image_hashes.get(unique_id)

        # 检测图片是否发生改变
        has_changed = current_hash != last_hash
        has_valid_images = images is not None

        # 决定使用模式：
        # - 如果没有输入图片，使用 cache 模式
        # - 如果有输入图片且内容发生改变，使用 input 模式
        # - 如果有输入图片但内容没有改变，使用 cache 模式
        if not has_valid_images:
            use_cache = True
            mode_reason = "No images input"
        elif has_changed:
            use_cache = False
            mode_reason = f"Images changed (hash: {last_hash} -> {current_hash})"
        else:
            use_cache = True
            mode_reason = f"Images unchanged (hash: {current_hash})"

        print(f"[CachePreviewBridge] Node {unique_id}: {mode_reason}, using {'cache' if use_cache else 'input'} mode")

        # 更新存储的hash值
        if current_hash is not None:
            preview_bridge_image_hashes[unique_id] = current_hash

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

class LG_LoadImage(LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                    },
                }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    DESCRIPTION = "从temp或output文件夹加载最新图片并复制到input文件夹。点击刷新按钮时，节点将更新图片列表并自动选择第一张图片，方便快速迭代。"
    CATEGORY = CATEGORY_TYPE
    FUNCTION = "load_image"

    @classmethod
    def IS_CHANGED(s, image):
        # 调用父类的IS_CHANGED方法
        return LoadImage.IS_CHANGED(image)

    def load_image(self, image):
        # 调用父类方法获取完整的图像和遮罩
        image_tensor, mask_tensor = super().load_image(image)
        
        # 返回图像、遮罩和文件名
        return (image_tensor, mask_tensor, image)


@PromptServer.instance.routes.get("/lg/get/latest_image")
async def get_latest_image(request):
    try:
        folder_type = request.query.get("type", "temp")
        
        if folder_type == "temp":
            target_dir = folder_paths.get_temp_directory()
        elif folder_type == "output":
            target_dir = folder_paths.get_output_directory()
        else:
            return web.json_response({"error": f"未知的文件夹类型: {folder_type}"}, status=400)
        
        files = [f for f in os.listdir(target_dir) 
                if os.path.isfile(os.path.join(target_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        if not files:
            return web.json_response({"error": f"在{folder_type}目录中未找到图像"}, status=404)
            
        latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(target_dir, f)))
        
        return web.json_response({
            "filename": latest_file,
            "subfolder": "",
            "type": folder_type
        })
    except Exception as e:
        print(f"[PreviewBridge] 获取最新图像出错: {str(e)}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

# 添加新的API路由，用于从temp/output复制图片到input
@PromptServer.instance.routes.post("/lg/copy_to_input")
async def copy_to_input(request):
    try:
        json_data = await request.json()
        folder_type = json_data.get("type", "temp")
        filename = json_data.get("filename")
        
        if not filename:
            return web.json_response({"error": "未指定文件名"}, status=400)
        
        # 确定源目录
        if folder_type == "temp":
            source_dir = folder_paths.get_temp_directory()
        elif folder_type == "output":
            source_dir = folder_paths.get_output_directory()
        else:
            return web.json_response({"error": f"未知的文件夹类型: {folder_type}"}, status=400)
        
        # 源文件完整路径
        source_path = os.path.join(source_dir, filename)
        if not os.path.exists(source_path):
            return web.json_response({"error": f"文件不存在: {source_path}"}, status=404)
        
        # 目标目录为input
        target_dir = folder_paths.get_input_directory()
        
        # 使用原始文件名
        target_filename = filename
        target_path = os.path.join(target_dir, target_filename)
        
        # 复制文件
        import shutil
        shutil.copy2(source_path, target_path)

        
        return web.json_response({
            "success": True,
            "filename": target_filename,
            "subfolder": "",
            "type": "input"
        })
    except Exception as e:
        print(f"[LG_LoadImage] 复制图片到input目录失败: {str(e)}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

# 在现有的API路由后添加删除文件的路由
@PromptServer.instance.routes.delete("/lg/delete_image")
async def delete_image(request):
    try:
        json_data = await request.json()
        filename = json_data.get("filename")
        
        if not filename:
            return web.json_response({"error": "未指定文件名"}, status=400)
        
        # 删除input目录中的文件
        input_dir = folder_paths.get_input_directory()
        file_path = os.path.join(input_dir, filename)
        
        if not os.path.exists(file_path):
            return web.json_response({"error": f"文件不存在: {filename}"}, status=404)
        
        # 删除文件
        os.remove(file_path)
        
        return web.json_response({
            "success": True,
            "message": f"文件 {filename} 已删除"
        })
    except Exception as e:
        print(f"[LG_LoadImage] 删除文件失败: {str(e)}")
        traceback.print_exc()
        return web.json_response({"error": str(e)}, status=500)

class LG_LatentBatchToList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    OUTPUT_IS_LIST = (True,)  # 表示输出是列表
    FUNCTION = "batch_to_list"
    CATEGORY = CATEGORY_TYPE
    
    def batch_to_list(self, latent):
        """将latent batch转换为latent列表"""
        samples = latent["samples"]
        batch_size = samples.shape[0]
        
        # 将batch分离为单独的latent
        latent_list = []
        for i in range(batch_size):
            single_latent = {"samples": samples[i:i+1]}  # 保持4维，但batch_size=1
            latent_list.append(single_latent)
        
        return (latent_list,)

class LG_PipManager:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "package_name": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "支持多种格式:\nnumpy==1.21.0  # 指定版本\ntorch>=1.9.0   # 版本范围\nrequests[security]  # 额外依赖\nnumpy torch pandas  # 多个包",
                    "tooltip": "包名支持版本号、额外依赖、多包安装等pip标准格式"
                }),
                "operation": (["install", "uninstall", "upgrade", "list"], {
                    "default": "install",
                    "tooltip": "选择操作类型：安装、卸载、升级或列出已安装包"
                }),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "manage_package"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_TYPE
    
    def manage_package(self, package_name, operation):
        import sys
        import re
        
        # 如果是 list 操作，忽略输入直接列出已安装包
        if operation == "list":
            cmd = [sys.executable, "-m", "pip", "list"]
            emoji = "📋"
            action = "列出已安装包"
            start_message = f"{emoji} 正在{action}..."
            
            # 使用通用方法执行命令
            success, result = execute_command_with_realtime_output(
                cmd, None, "/lg/pip_manager", True, start_message
            )
            
            if isinstance(result, str):  # 中断或异常
                return {"ui": {"text": (result,)}}
            
            if success:
                success_msg = "✅ 已列出所有已安装的包"
                PromptServer.instance.send_sync("/lg/pip_manager", {
                    "text": success_msg,
                    "clear": False
                })
                return {"ui": {"text": (success_msg,)}}
            else:
                error_msg = f"❌ 列出包失败 (退出码: {result})"
                PromptServer.instance.send_sync("/lg/pip_manager", {
                    "text": error_msg,
                    "clear": False
                })
                return {"ui": {"text": (error_msg,)}}
        
        # 其他操作需要检查包名输入
        if not package_name.strip():
            return {"ui": {"text": ("❌ 请输入包名",)}}
        
        # 处理多行输入，过滤注释和空行
        lines = []
        for line in package_name.strip().split('\n'):
            line = line.split('#')[0].strip()  # 移除注释
            if line:  # 忽略空行
                lines.append(line)
        
        if not lines:
            return {"ui": {"text": ("❌ 请输入有效的包名",)}}
        
        # 构建pip命令
        if operation == "install":
            cmd = [sys.executable, "-m", "pip", "install"] + lines
            emoji = "📦"
            action = "安装"
            packages_text = " ".join(lines)
        elif operation == "uninstall":
            # 对于卸载，需要提取基础包名（去掉版本号和额外依赖）
            base_packages = []
            for line in lines:
                for pkg in line.split():
                    # 提取基础包名：移除版本号、额外依赖等
                    base_name = re.split(r'[><=!~\[]', pkg)[0].strip()
                    if base_name:
                        base_packages.append(base_name)
            
            if not base_packages:
                return {"ui": {"text": ("❌ 无法提取有效的包名",)}}
            
            cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + base_packages
            emoji = "🗑️"
            action = "卸载"
            packages_text = " ".join(base_packages)
        else:  # upgrade
            cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + lines
            emoji = "⬆️"
            action = "升级"
            packages_text = " ".join(lines)
        
        # 添加卸载警告
        warning_msg = ""
        if operation == "uninstall":
            warning_msg = "\n⚠️  注意：如果包正在使用中，卸载可能失败。建议重启ComfyUI后再尝试。"
        
        start_message = f"{emoji} 开始{action} {packages_text}...{warning_msg}\n📝 执行命令: {' '.join(cmd)}"
        
        # 使用通用方法执行命令
        success, result = execute_command_with_realtime_output(
            cmd, None, "/lg/pip_manager", True, start_message
        )
        
        if isinstance(result, str):  # 中断或异常
            return {"ui": {"text": (result,)}}
        
        if success:
            success_msg = f"✅ 成功{action} {packages_text}"
            PromptServer.instance.send_sync("/lg/pip_manager", {
                "text": success_msg,
                "clear": False
            })
            return {"ui": {"text": (success_msg,)}}
        else:
            error_msg = f"❌ {action}失败 (退出码: {result})"
            PromptServer.instance.send_sync("/lg/pip_manager", {
                "text": error_msg,
                "clear": False
            })
            return {"ui": {"text": (error_msg,)}}

class LG_SaveImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {
                    "default": "ComfyUI_{timestamp}", 
                    "multiline": False,
                    "tooltip": "文件名前缀，支持表达式：{timestamp}时间戳、{date}日期、{time}时间、{datetime}日期时间、{batch}批次号、{counter}计数器"
                }),
                "path": ("STRING", {
                    "default": "", 
                    "multiline": False, 
                    "placeholder": "留空使用默认输出目录",
                    "tooltip": "保存路径，支持绝对路径和相对路径，不存在时自动创建"
                }),
                "format": (["png", "jpg", "webp"], {
                    "default": "png",
                    "tooltip": "图像保存格式：PNG无损、JPG/WebP有损压缩"
                }),
                "quality": ("INT", {
                    "default": 95, 
                    "min": 1, 
                    "max": 100, 
                    "step": 1,
                    "tooltip": "图像质量(1-100)，仅对JPG和WebP格式有效，PNG格式忽略此参数"
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_TYPE

    def save_images(self, images, filename_prefix="ComfyUI_{timestamp}", path="", format="png", quality=95):
        # 确定保存路径
        if path:
            # 支持相对路径和绝对路径
            if os.path.isabs(path):
                save_dir = path
            else:
                # 相对路径基于ComfyUI根目录
                save_dir = os.path.join(os.getcwd(), path)
        else:
            # 使用默认输出目录
            save_dir = folder_paths.get_output_directory()
        
        # 创建目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)
        
        # 一次性获取时间信息（避免重复计算）
        import datetime
        now = datetime.datetime.now()
        timestamp = str(int(now.timestamp()))
        date_str = now.strftime("%Y%m%d")
        time_str = now.strftime("%H%M%S")
        datetime_str = now.strftime("%Y%m%d_%H%M%S")
        
        # 预处理文件名前缀，只替换非批次相关的变量
        base_prefix = filename_prefix.replace("{timestamp}", timestamp)
        base_prefix = base_prefix.replace("{date}", date_str)
        base_prefix = base_prefix.replace("{time}", time_str)
        base_prefix = base_prefix.replace("{datetime}", datetime_str)
        
        file_extension = f".{format}"
        
        # 使用类似系统的计数器逻辑
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(base_prefix, save_dir, images[0].shape[1], images[0].shape[0])
        
        for batch_number, image in enumerate(images):
            # 转换tensor为PIL图像
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # 处理批次和计数器变量
            processed_prefix = base_prefix
            if "{batch}" in filename_prefix:
                processed_prefix = processed_prefix.replace("{batch}", f"{batch_number:05d}")
            if "{counter}" in filename_prefix:
                processed_prefix = processed_prefix.replace("{counter}", f"{counter:05d}")
            
            # 生成文件名
            final_filename = f"{processed_prefix}{file_extension}"
            
            # 如果没有使用批次或计数器变量，且有多张图片，需要避免重名
            if len(images) > 1 and "{batch}" not in filename_prefix and "{counter}" not in filename_prefix:
                name_without_ext = os.path.splitext(final_filename)[0]
                final_filename = f"{name_without_ext}_{batch_number:05d}{file_extension}"
            
            file_path = os.path.join(full_output_folder, final_filename)
            counter += 1
            
            # 根据格式保存图像，移除optimize减少处理时间
            if format == "png":
                # 使用与系统相同的compress_level
                img.save(file_path, format='PNG', compress_level=4)

            elif format == "jpg":
                # 确保RGB模式（JPEG不支持透明度）
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(file_path, format='JPEG', quality=quality)

            elif format == "webp":
                img.save(file_path, format='WebP', quality=quality)

        
        return {}

class LG_InstallDependencies:
    @classmethod
    def INPUT_TYPES(s):
        # 获取custom_nodes目录
        custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes")
        custom_nodes_dir = custom_nodes_paths[0] if custom_nodes_paths else None
        
        # 获取所有包含requirements.txt的插件文件夹
        plugin_folders = []
        if custom_nodes_dir and os.path.exists(custom_nodes_dir):
            for item in os.listdir(custom_nodes_dir):
                item_path = os.path.join(custom_nodes_dir, item)
                if os.path.isdir(item_path):
                    requirements_path = os.path.join(item_path, "requirements.txt")
                    if os.path.exists(requirements_path):
                        plugin_folders.append(item)
        
        # 如果没有找到包含requirements.txt的插件，添加一个提示项
        if not plugin_folders:
            plugin_folders = ["未找到包含requirements.txt的插件"]
        
        return {
            "required": {
                "plugin_folder": (sorted(plugin_folders), {"tooltip": "选择要安装依赖的插件文件夹"}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "install_dependencies"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_TYPE
    
    def install_dependencies(self, plugin_folder):
        if plugin_folder == "未找到包含requirements.txt的插件":
            return {"ui": {"text": ("错误: 没有找到包含requirements.txt的插件文件夹",)}}
        
        # 获取custom_nodes目录和插件路径
        custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes")
        custom_nodes_dir = custom_nodes_paths[0] if custom_nodes_paths else None
        
        if not custom_nodes_dir:
            return {"ui": {"text": ("❌ 无法获取custom_nodes目录路径",)}}
        
        plugin_path = os.path.join(custom_nodes_dir, plugin_folder)
        requirements_path = os.path.join(plugin_path, "requirements.txt")
        
        if not os.path.exists(plugin_path):
            return {"ui": {"text": (f"❌ 插件目录不存在",)}}
        
        if not os.path.exists(requirements_path):
            return {"ui": {"text": (f"❌ requirements.txt不存在",)}}
        
        import sys
        
        # 构建pip命令
        cmd = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        start_message = f"🚀 开始安装 {plugin_folder} 的依赖包...\n📝 执行命令: {' '.join(cmd)}"
        
        # 使用通用方法执行命令
        success, result = execute_command_with_realtime_output(
            cmd, plugin_path, "/lg/install_dependencies", True, start_message
        )
        
        if isinstance(result, str):  # 中断或异常
            return {"ui": {"text": (result,)}}
        
        if success:
            success_msg = f"✅ 成功安装 {plugin_folder} 的依赖包"
            PromptServer.instance.send_sync("/lg/install_dependencies", {
                "text": success_msg,
                "clear": False
            })
            return {"ui": {"text": (success_msg,)}}
        else:
            error_msg = f"❌ 安装失败 (退出码: {result})"
            PromptServer.instance.send_sync("/lg/install_dependencies", {
                "text": error_msg,
                "clear": False
            })
            return {"ui": {"text": (error_msg,)}}

class LG_FloatRange:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.00, 
                    "min": 0.00, 
                    "max": 1.00, 
                    "step": 0.01,
                    "round": 0.01,
                    "tooltip": "0-1范围的浮点数，精度控制在小数点后2位"
                }),
            }
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_float_value"
    CATEGORY = CATEGORY_TYPE
    
    def get_float_value(self, value):
        rounded_value = round(value, 2)
        return (rounded_value,)

NODE_CLASS_MAPPINGS = {
    "CachePreviewBridge": CachePreviewBridge,
    "LG_Noise": LG_Noise,
    "IPAdapterWeightTypes": IPAdapterWeightTypes,
    "LG_LoadImage": LG_LoadImage,
    "LG_LatentBatchToList": LG_LatentBatchToList,
    "LG_SaveImage": LG_SaveImage,
    "LG_InstallDependencies": LG_InstallDependencies,
    "LG_PipManager": LG_PipManager,
    "LG_FloatRange": LG_FloatRange,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CachePreviewBridge": "🎈LG_PreviewBridge",
    "LG_Noise": "🎈LG_Noise",
    "IPAdapterWeightTypes": "🎈IPAdapter权重类型",
    "LG_LoadImage": "🎈LG_LoadImage",
    "LG_LatentBatchToList": "🎈LG_Latent批次转列表",
    "LG_SaveImage": "🎈LG_SaveImage",
    "LG_InstallDependencies": "🎈LG_安装依赖",
    "LG_PipManager": "🎈LG_Pip管理器",
    "LG_FloatRange": "🎈LG_浮点数[0-1]",
}



