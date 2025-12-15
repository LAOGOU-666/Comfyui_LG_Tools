from .md import *
import json

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

# 缓存相关的全局变量
preview_bridge_cache = {}
preview_bridge_image_hashes = {}


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
    def load_image_from_fileinfo(file_info_json):
        """从 JSON 文件信息或字符串路径加载图片"""
        # 初始化默认值（64*64 遮罩表示无效遮罩）
        image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        ui_item = {
            "filename": 'empty.png',
            "subfolder": '',
            "type": 'temp'
        }
        
        if not file_info_json:
            final_mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
            return image, final_mask, ui_item
        
        filename = None
        subfolder = ''
        file_type = 'input'
        
        try:
            # 首先尝试作为 JSON 解析（旧版本格式）
            file_info = json.loads(file_info_json)
            filename = file_info.get('filename')
            subfolder = file_info.get('subfolder', '')
            file_type = file_info.get('type', 'input')
            
        except json.JSONDecodeError:
            # JSON 解析失败，尝试解析新版本字符串格式
            # 新版本格式: "path/filename.ext [type]" 或 "filename.ext [type]"
            file_info_str = str(file_info_json).strip()
            
            # 查找最后的 [type] 部分
            if '[' in file_info_str and file_info_str.endswith(']'):
                # 分离路径和类型
                last_bracket = file_info_str.rfind('[')
                path_part = file_info_str[:last_bracket].strip()
                type_part = file_info_str[last_bracket+1:-1].strip()
                
                # 解析路径部分
                if '/' in path_part:
                    # 包含子文件夹
                    path_parts = path_part.split('/')
                    filename = path_parts[-1]  # 最后一部分是文件名
                    subfolder = '/'.join(path_parts[:-1])  # 前面的部分是子文件夹
                else:
                    # 只有文件名
                    filename = path_part
                    subfolder = ''
                
                # 解析类型
                file_type = type_part.lower() if type_part else 'input'
            else:
                # 没有类型信息，直接作为路径处理
                if '/' in file_info_str:
                    path_parts = file_info_str.split('/')
                    filename = path_parts[-1]
                    subfolder = '/'.join(path_parts[:-1])
                else:
                    filename = file_info_str
                    subfolder = ''
                file_type = 'input'  # 默认类型
        
        if not filename:
            final_mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
            return image, final_mask, ui_item
        
        try:
            # 构建文件路径
            if file_type == 'input':
                base_dir = folder_paths.get_input_directory()
            elif file_type == 'output':
                base_dir = folder_paths.get_output_directory()
            elif file_type == 'temp':
                base_dir = folder_paths.get_temp_directory()
            else:
                base_dir = folder_paths.get_input_directory()
            
            if subfolder:
                file_path = os.path.join(base_dir, subfolder, filename)
            else:
                file_path = os.path.join(base_dir, filename)
            
            # 更新 ui_item
            ui_item = {
                "filename": filename,
                "subfolder": subfolder,
                "type": file_type
            }
            
            # 加载图片
            if os.path.exists(file_path):
                i = Image.open(file_path)
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
            print(f"[CachePreviewBridge] 加载图片失败: {e}")

        final_mask = mask.unsqueeze(0) if len(mask.shape) == 2 else mask
        return image, final_mask, ui_item

    def doit(self, image, unique_id, images=None, extra_pnginfo=None):
        """
        简化逻辑：
        1. 如果有 images 输入，保存并返回
        2. 如果没有 images 输入但有 image (文件信息)，从文件加载
        3. 否则返回空图像
        """
        # 情况1: 有图像输入 - 从 clipspace 加载遮罩，或使用输入的图像
        if images is not None:
            # 更新 hash
            current_hash = self.calculate_image_hash(images)
            preview_bridge_image_hashes[unique_id] = current_hash
            
            # 默认遮罩（64*64 表示无效遮罩）
            mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            ui_item = None
            
            # 如果有文件信息，尝试从文件加载遮罩
            if image and image.strip():
                _, loaded_mask, path_item = CachePreviewBridge.load_image_from_fileinfo(image)
                
                # 如果成功加载了遮罩，使用它
                if path_item["filename"] != 'empty.png':
                    # 调整遮罩尺寸以匹配图像
                    if loaded_mask.shape[-2:] == (images.shape[1], images.shape[2]):
                        mask = loaded_mask
                        ui_item = path_item  # 保存 clipspace 文件信息用于 UI 预览
            
            # 如果有 clipspace 文件，直接返回 clipspace 的预览
            if ui_item:
                return {
                    "ui": {"images": [ui_item]},
                    "result": (images, mask),
                }
            
            # 否则保存新预览图并返回默认的 64*64 遮罩
            res = PreviewImage().save_images(
                images, 
                filename_prefix=f"PreviewBridge/PB-{unique_id}-", 
                extra_pnginfo=extra_pnginfo
            )
            
            # 缓存结果
            preview_bridge_cache[unique_id] = (images, res['ui']['images'])
            
            return {
                "ui": {"images": res['ui']['images']},
                "result": (images, mask),
            }
        
        # 情况2: 没有图像输入，但有文件信息 - 从文件加载
        if image and image.strip():
            pixels, mask, path_item = CachePreviewBridge.load_image_from_fileinfo(image)
            
            # 检查是否成功加载
            if path_item["filename"] != 'empty.png':
                return {
                    "ui": {"images": [path_item]},
                    "result": (pixels, mask),
                }
        
        # 情况3: 检查是否有缓存
        if unique_id in preview_bridge_cache:
            cached_images, cached_ui = preview_bridge_cache[unique_id]
            # 返回 64*64 遮罩表示无效遮罩
            mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            return {
                "ui": {"images": cached_ui},
                "result": (cached_images, mask),
            }
        
        # 情况4: 没有任何数据 - 返回空图像和 64*64 遮罩
        empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
        empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
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


# API路由：删除文件
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



# 计数器节点状态存储
counter_states = {}

# 图片加载器计数器状态存储
image_loader_counter_states = {}

class LG_ImageLoaderWithCounter:
    """
    带计数器的图片加载器节点
    从文件夹加载图片，内置计数器功能和多种排序模式
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "图片文件夹路径（支持相对路径和绝对路径）"
                }),
                "mode": (["increase", "decrease", "all"], {
                    "default": "increase",
                    "tooltip": "increase: 递增模式，decrease: 递减模式，all: 加载所有图像"
                }),
                "sort_mode": (["Alphabetical (ASC)", "Alphabetical (DESC)", "Numerical (ASC)", "Numerical (DESC)", "Datetime (ASC)", "Datetime (DESC)"], {
                    "default": "Alphabetical (ASC)",
                    "tooltip": "图片排序方式"
                }),
                "keep_index": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "开启时保持当前索引不变，关闭时恢复正常递增/递减"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "INT")
    RETURN_NAMES = ("images", "masks", "filenames", "current_index", "total_images")
    OUTPUT_IS_LIST = (True, True, True, False, False)
    FUNCTION = "load_image_with_counter"
    CATEGORY = CATEGORY_TYPE
    DESCRIPTION = "从文件夹加载图片，内置计数器自动获取文件夹总数，支持递增/递减/全部加载模式和多种排序方式"
    
    @classmethod
    def IS_CHANGED(s, folder_path, mode, sort_mode, keep_index, unique_id):
        """判断节点是否需要重新执行"""
        # all模式或keep_index开启时不触发自动更新，只在参数改变时更新
        if mode == "all" or keep_index:
            return f"{folder_path}_{sort_mode}_{keep_index}"
        # 其他模式每次都返回不同的浮点数，确保每次都触发执行
        import time
        return float(time.time())
    
    def get_image_files(self, folder_path, sort_mode):
        """
        获取文件夹中的所有图片文件并排序
        
        Args:
            folder_path: 文件夹路径
            sort_mode: 排序模式
            
        Returns:
            排序后的图片文件列表
        """
        import re
        
        # 支持的图片格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}
        
        # 处理路径
        if not os.path.isabs(folder_path):
            # 相对路径：相对于ComfyUI根目录
            folder_path = os.path.join(os.getcwd(), folder_path)
        
        if not os.path.exists(folder_path):
            print(f"[ImageLoaderCounter] 文件夹不存在: {folder_path}")
            return []
        
        # 获取所有图片文件
        files = []
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in image_extensions:
                    files.append(f)
        
        if not files:
            print(f"[ImageLoaderCounter] 文件夹中没有图片: {folder_path}")
            return []
        
        # 根据排序模式排序
        if sort_mode == "Alphabetical (ASC)":
            files.sort()
        elif sort_mode == "Alphabetical (DESC)":
            files.sort(reverse=True)
        elif sort_mode == "Numerical (ASC)":
            # 提取数字排序
            def extract_number(filename):
                numbers = re.findall(r'\d+', filename)
                return int(numbers[0]) if numbers else 0
            files.sort(key=extract_number)
        elif sort_mode == "Numerical (DESC)":
            def extract_number(filename):
                numbers = re.findall(r'\d+', filename)
                return int(numbers[0]) if numbers else 0
            files.sort(key=extract_number, reverse=True)
        elif sort_mode == "Datetime (ASC)":
            # 按文件修改时间排序
            files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)))
        elif sort_mode == "Datetime (DESC)":
            files.sort(key=lambda f: os.path.getmtime(os.path.join(folder_path, f)), reverse=True)
        
        return files
    
    def load_image_with_counter(self, folder_path, mode, sort_mode, keep_index, unique_id):
        """
        加载图片并应用计数器逻辑
        """
        try:
            # 处理路径
            if not os.path.isabs(folder_path):
                folder_path = os.path.join(os.getcwd(), folder_path)
            
            # 获取排序后的图片列表
            image_files = self.get_image_files(folder_path, sort_mode)
            total_images = len(image_files)
            
            if total_images == 0:
                print(f"[ImageLoaderCounter] 文件夹中没有图片")
                # 返回空图像列表
                empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
                return ([empty_image], [empty_mask], [""], 0, 0)
            
            # all模式：加载所有图片
            if mode == "all":
                print(f"[ImageLoaderCounter] 节点 {unique_id} all模式，加载所有图片: {total_images}张")
                
                images_list = []
                masks_list = []
                filenames_list = []
                
                for idx, filename in enumerate(image_files):
                    image_path = os.path.join(folder_path, filename)
                    
                    # 使用PIL加载图片
                    img = Image.open(image_path)
                    img = ImageOps.exif_transpose(img)
                    
                    # 转换为RGB
                    if img.mode == 'I':
                        img = img.point(lambda i: i * (1 / 255))
                    image = img.convert("RGB")
                    
                    # 转换为tensor
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None,]
                    
                    # 处理mask
                    if 'A' in img.getbands():
                        mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                        mask = 1. - torch.from_numpy(mask)
                    else:
                        mask = torch.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32, device="cpu")
                    
                    # 确保mask是正确的维度
                    if len(mask.shape) == 2:
                        mask = mask.unsqueeze(0)
                    
                    images_list.append(image_tensor)
                    masks_list.append(mask)
                    # 移除文件扩展名
                    filename_without_ext = os.path.splitext(filename)[0]
                    filenames_list.append(filename_without_ext)

                    print(f"[ImageLoaderCounter] 加载 {idx+1}/{total_images}: {filename}")
                
                # 发送状态到前端
                PromptServer.instance.send_sync("counter_update", {
                    "node_id": unique_id,
                    "count": total_images,
                    "total": total_images
                })
                
                return (images_list, masks_list, filenames_list, total_images, total_images)
            
            # 确定当前索引（递增/递减模式）
            current_index = 0
            
            # 初始化或获取当前节点的计数状态
            if unique_id not in image_loader_counter_states:
                image_loader_counter_states[unique_id] = {
                    "current": 0 if mode == "increase" else total_images - 1,
                    "total": total_images,
                    "mode": mode,
                    "sort_mode": sort_mode,
                    "folder_path": folder_path,
                    "first_run": True
                }
                print(f"[ImageLoaderCounter] 初始化节点 {unique_id}, 模式: {mode}, 总数: {total_images}")
            
            state = image_loader_counter_states[unique_id]
            
            # 如果参数改变了，重置计数器
            if (state["total"] != total_images or 
                state["mode"] != mode or 
                state["sort_mode"] != sort_mode or
                state["folder_path"] != folder_path):
                state["total"] = total_images
                state["mode"] = mode
                state["sort_mode"] = sort_mode
                state["folder_path"] = folder_path
                state["current"] = 0 if mode == "increase" else total_images - 1
                state["first_run"] = True
                print(f"[ImageLoaderCounter] 参数改变，重置节点 {unique_id}, 新模式: {mode}, 新总数: {total_images}")
            
            current_index = state["current"]
            
            # 如果开启了keep_index，不更新计数器，保持当前索引
            if keep_index:
                print(f"[ImageLoaderCounter] 节点 {unique_id} keep_index开启，保持索引: {current_index}/{total_images}")
            # 如果不是第一次运行且没有开启keep_index，则更新计数器
            elif not state["first_run"]:
                if mode == "increase":
                    # 递增模式：0 -> total-1, 然后循环回0
                    state["current"] += 1
                    if state["current"] >= total_images:
                        state["current"] = 0
                else:  # decrease
                    # 递减模式：total-1 -> 0, 然后循环回total-1
                    state["current"] -= 1
                    if state["current"] < 0:
                        state["current"] = total_images - 1
                
                current_index = state["current"]
                print(f"[ImageLoaderCounter] 节点 {unique_id} 执行, 返回索引: {current_index}, 下次: {state['current']}")
            else:
                # 第一次运行，标记为非首次
                state["first_run"] = False
                print(f"[ImageLoaderCounter] 节点 {unique_id} 首次执行, 索引: {current_index}/{total_images}")
            
            # 加载对应索引的图片
            filename = image_files[current_index]
            image_path = os.path.join(folder_path, filename)
            
            # 使用PIL加载图片
            img = Image.open(image_path)
            img = ImageOps.exif_transpose(img)
            
            # 转换为RGB
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            image = img.convert("RGB")
            
            # 转换为tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            # 处理mask
            if 'A' in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.float32, device="cpu")
            
            # 确保mask是正确的维度
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            
            print(f"[ImageLoaderCounter] 加载图片: {filename}, 索引: {current_index}/{total_images-1} (第{current_index+1}/{total_images}张)")

            # 发送状态到前端，显示"第几张/总数"（索引+1）
            PromptServer.instance.send_sync("counter_update", {
                "node_id": unique_id,
                "count": current_index + 1,  # 显示第几张（1-based）
                "total": total_images
            })
            
            # 返回列表格式（与all模式保持一致）
            # 移除文件扩展名
            filename_without_ext = os.path.splitext(filename)[0]
            return ([image_tensor], [mask], [filename_without_ext], current_index, total_images)
            
        except Exception as e:
            print(f"[ImageLoaderCounter] 加载图片失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回空图像列表
            empty_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32, device="cpu")
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
            return ([empty_image], [empty_mask], [""], 0, 0)

class LG_Counter:
    """计数器节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "total": ("INT", {
                    "default": 10, 
                    "min": 1, 
                    "max": 10000,
                    "step": 1
                }),
                "mode": (["increase", "decrease", "None"], {
                    "default": "increase"
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("count",)
    FUNCTION = "count"
    CATEGORY = CATEGORY_TYPE
    
    @classmethod
    def IS_CHANGED(s, total, mode, unique_id):
        # None模式不触发自动更新，只在参数改变时更新
        if mode == "None":
            return f"{total}_{mode}"
        # 其他模式每次都返回不同的浮点数，确保每次都触发执行
        import time
        return float(time.time())

    def count(self, total, mode, unique_id):
        try:
            # None模式：直接返回total-1的索引值，不进行计数
            if mode == "None":
                index_value = total - 1
                print(f"[Counter] 节点 {unique_id} None模式，返回索引值: {index_value} (total={total})")
                # 发送计数到前端
                PromptServer.instance.send_sync("counter_update", {
                    "node_id": unique_id,
                    "count": index_value
                })
                return (index_value,)
            
            # 初始化或获取当前节点的计数状态
            if unique_id not in counter_states:
                counter_states[unique_id] = {
                    "current": 0 if mode == "increase" else total,
                    "total": total,
                    "mode": mode,
                    "first_run": True
                }
                print(f"[Counter] 初始化节点 {unique_id}, 模式: {mode}, 总数: {total}")
            
            state = counter_states[unique_id]
            
            # 如果total或mode改变了，重置计数器
            if state["total"] != total or state["mode"] != mode:
                state["total"] = total
                state["mode"] = mode
                state["current"] = 0 if mode == "increase" else total
                state["first_run"] = True
                print(f"[Counter] 参数改变，重置节点 {unique_id}, 新模式: {mode}, 新总数: {total}")
            
            current_count = state["current"]
            
            # 如果不是第一次运行，则先更新计数器再返回
            if not state["first_run"]:
                if mode == "increase":
                    # 递增模式：0 -> total-1, 然后循环回0
                    state["current"] += 1
                    if state["current"] >= total:
                        state["current"] = 0
                else:
                    # 递减模式：total -> 0, 然后循环回total
                    state["current"] -= 1
                    if state["current"] < 0:
                        state["current"] = total
                
                current_count = state["current"]
            else:
                # 第一次运行，标记为非首次
                state["first_run"] = False
            
            print(f"[Counter] 节点 {unique_id} 执行, 返回: {current_count}, 下次: {state['current']}")
            
            # 发送计数到前端
            PromptServer.instance.send_sync("counter_update", {
                "node_id": unique_id,
                "count": current_count
            })
            
            return (current_count,)
            
        except Exception as e:
            print(f"[Counter] 计数器执行错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return (0,)

# API端点：重置计数器
@routes.post('/counter/reset')
async def reset_counter(request):
    try:
        data = await request.json()
        node_id = data.get('node_id')
        
        if not node_id:
            return web.json_response({"status": "error", "message": "节点ID不能为空"}, status=400)
        
        print(f"[Counter] 重置请求 - node_id: {node_id}, 类型: {type(node_id)}")
        print(f"[Counter] 当前存储的keys: {list(counter_states.keys())}")
        
        # 重置计数器状态 - 检查字符串和整数格式
        found = False
        target_key = None
        
        # 先直接查找
        if node_id in counter_states:
            target_key = node_id
            found = True
        # 尝试整数格式
        elif str(node_id) in counter_states:
            target_key = str(node_id)
            found = True
        # 尝试字符串转整数
        else:
            try:
                int_id = int(node_id)
                if int_id in counter_states:
                    target_key = int_id
                    found = True
            except (ValueError, TypeError):
                pass
        
        if found:
            state = counter_states[target_key]
            if state["mode"] == "increase":
                state["current"] = 0
            else:
                state["current"] = state["total"]
            
            state["first_run"] = True
            
            print(f"[Counter] 重置成功 - 当前值: {state['current']}")
            
            return web.json_response({
                "status": "success", 
                "current": state["current"],
                "message": "计数器已重置"
            })
        else:
            print(f"[Counter] 未找到计数器状态")
            return web.json_response({
                "status": "success",
                "message": "计数器状态不存在，将在下次执行时初始化"
            })
            
    except Exception as e:
        print(f"[Counter] 重置计数器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return web.json_response({"status": "error", "message": str(e)}, status=500)

# API端点：重置图片加载器计数器
@routes.post('/image_loader_counter/reset')
async def reset_image_loader_counter(request):
    """
    重置图片加载器计数器状态
    """
    try:
        data = await request.json()
        node_id = data.get('node_id')
        
        if not node_id:
            return web.json_response({"status": "error", "message": "节点ID不能为空"}, status=400)
        
        print(f"[ImageLoaderCounter] 重置请求 - node_id: {node_id}, 类型: {type(node_id)}")
        print(f"[ImageLoaderCounter] 当前存储的keys: {list(image_loader_counter_states.keys())}")
        
        # 重置计数器状态 - 检查字符串和整数格式
        found = False
        target_key = None
        
        # 先直接查找
        if node_id in image_loader_counter_states:
            target_key = node_id
            found = True
        # 尝试整数格式
        elif str(node_id) in image_loader_counter_states:
            target_key = str(node_id)
            found = True
        # 尝试字符串转整数
        else:
            try:
                int_id = int(node_id)
                if int_id in image_loader_counter_states:
                    target_key = int_id
                    found = True
            except (ValueError, TypeError):
                pass
        
        if found:
            state = image_loader_counter_states[target_key]
            if state["mode"] == "increase":
                state["current"] = 0
            else:
                state["current"] = state["total"] - 1
            
            state["first_run"] = True
            
            print(f"[ImageLoaderCounter] 重置成功 - 当前索引: {state['current']}/{state['total']}")
            
            return web.json_response({
                "status": "success", 
                "current": state["current"],
                "total": state["total"],
                "message": "图片加载器计数器已重置"
            })
        else:
            print(f"[ImageLoaderCounter] 未找到计数器状态")
            return web.json_response({
                "status": "success",
                "current": 0,
                "message": "计数器状态不存在，将在下次执行时初始化"
            })
            
    except Exception as e:
        print(f"[ImageLoaderCounter] 重置计数器失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return web.json_response({"status": "error", "message": str(e)}, status=500)

# 存储每个节点的基准图片信息
loadimage_baseline = {}

class LG_LoadImage_V2(LoadImage):
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True}),
                     "auto_refresh": ("BOOLEAN", {"default": True}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "filename")
    DESCRIPTION = "加载图片节点。auto_refresh开启后，自动加载输入文件夹的最新图片。"
    CATEGORY = CATEGORY_TYPE
    FUNCTION = "load_image"

    @classmethod
    def IS_CHANGED(s, image, auto_refresh, unique_id):
        if auto_refresh:
            # 在auto_refresh模式下，返回浮点数确保每次更新
            import time
            return float(time.time())
        # 否则调用父类的IS_CHANGED方法
        return LoadImage.IS_CHANGED(image)

    def load_image(self, image, auto_refresh, unique_id):
        input_dir = folder_paths.get_input_directory()
        
        # 如果auto_refresh开启，执行智能加载逻辑
        if auto_refresh:
            # 获取当前传入的图片路径和时间戳
            current_image_path = os.path.join(input_dir, image)
            
            if os.path.exists(current_image_path):
                current_timestamp = os.path.getmtime(current_image_path)
                
                # 初始化：第一次运行
                if unique_id not in loadimage_baseline:
                    loadimage_baseline[unique_id] = {
                        "image": image,
                        "timestamp": current_timestamp,
                        "last_input_image": image  # 记录上次前端传入的image参数
                    }
                    print(f"[LG_LoadImage] 节点 {unique_id} 初始化基准图片: {image}, 时间戳: {current_timestamp}")
                else:
                    # 检查用户是否在前端手动选择了新图片
                    # 比较传入的image和上次传入的image参数（来自前端）
                    last_input_image = loadimage_baseline[unique_id].get("last_input_image", loadimage_baseline[unique_id]["image"])
                    
                    if image != last_input_image:
                        # 前端的image参数变了，说明用户手动选择了新图片，更新基准
                        loadimage_baseline[unique_id] = {
                            "image": image,
                            "timestamp": current_timestamp,
                            "last_input_image": image
                        }
                        print(f"[LG_LoadImage] 节点 {unique_id} 用户手动选择新图片: {image}, 时间戳: {current_timestamp}")
                    else:
                        # 前端的image参数没变，执行自动加载逻辑
                        baseline_timestamp = loadimage_baseline[unique_id]["timestamp"]
                        
                        # 获取所有图片文件
                        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
                        files = folder_paths.filter_files_content_types(files, ["image"])
                        
                        if files:
                            # 找到所有比基准图片更新的图片
                            newer_files = [f for f in files if os.path.getmtime(os.path.join(input_dir, f)) > baseline_timestamp]
                            
                            if newer_files:
                                # 从更新的图片中选择最新的
                                latest_file = max(newer_files, key=lambda f: os.path.getmtime(os.path.join(input_dir, f)))
                                latest_timestamp = os.path.getmtime(os.path.join(input_dir, latest_file))
                                
                                # 更新基准为新加载的图片（但保持last_input_image不变）
                                loadimage_baseline[unique_id]["image"] = latest_file
                                loadimage_baseline[unique_id]["timestamp"] = latest_timestamp
                                # last_input_image 保持不变，因为前端传入的image参数没变
                                
                                image = latest_file
                                print(f"[LG_LoadImage] 节点 {unique_id} 自动加载更新的图片: {image}, 时间戳: {latest_timestamp}")
                            else:
                                print(f"[LG_LoadImage] 节点 {unique_id} 没有比基准更新的图片，继续使用基准: {loadimage_baseline[unique_id]['image']}")
                                # 使用基准图片
                                image = loadimage_baseline[unique_id]["image"]
        
        # 调用父类方法获取完整的图像和遮罩
        image_tensor, mask_tensor = super().load_image(image)
        
        # 返回图像、遮罩和文件名
        return (image_tensor, mask_tensor, image)
    
class LG_MaskHoleFiller:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),  # 输入维度: (B, H, W)
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("filled_mask",)
    FUNCTION = "fill_mask_holes"
    CATEGORY = "mask"

    def fill_mask_holes(self, mask):
        filled_masks = []
        for i in range(mask.shape[0]):
            mask_np = (mask[i].cpu().numpy() > 0.5).astype(np.uint8)
            h, w = mask_np.shape

            # 在遮罩周围添加1像素的边框（值为0），确保外部区域连通
            padded_mask = np.pad(mask_np, pad_width=1, mode='constant', constant_values=0)

            # 创建floodFill所需的mask，必须比padded_mask大2
            floodfill = padded_mask.copy()
            mask_ff = np.zeros((h + 4, w + 4), np.uint8)

            # 从(0,0)点进行floodFill，填充外部区域
            cv2.floodFill(floodfill, mask_ff, (0, 0), 1)

            # floodfill==0的地方是内部的空洞
            holes = (floodfill == 0).astype(np.uint8)

            # 移除边框，恢复原始尺寸
            holes = holes[1:-1, 1:-1]

            # 原mask加上空洞部分
            filled = np.clip(mask_np + holes, 0, 1)
            filled_tensor = torch.from_numpy(filled).float()
            filled_masks.append(filled_tensor)

        filled_mask = torch.stack(filled_masks)
        return (filled_mask,)

class LG_MaskBatchMerge:
    """合并批次遮罩节点 - 将批次中的多个遮罩合并为单个遮罩"""

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),  # 输入维度: (B, H, W)
                "merge_mode": (["max", "min", "average", "sum"], {
                    "default": "max"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("merged_mask",)
    FUNCTION = "merge_masks"
    CATEGORY = CATEGORY_TYPE
    DESCRIPTION = "合并批次遮罩为单个遮罩。支持最大值、最小值、平均值和求和四种合并模式。"

    def merge_masks(self, masks, merge_mode):
        """
        合并批次遮罩

        参数:
            masks: 输入的批次遮罩，形状为 (B, H, W)
            merge_mode: 合并模式
                - "max": 取所有遮罩的最大值（并集）
                - "min": 取所有遮罩的最小值（交集）
                - "average": 取所有遮罩的平均值
                - "sum": 取所有遮罩的和（会被裁剪到0-1范围）

        返回:
            merged_mask: 合并后的遮罩，形状为 (1, H, W)
        """
        if masks.shape[0] == 0:
            raise ValueError("输入的遮罩批次为空")

        # 根据合并模式处理
        if merge_mode == "max":
            # 取最大值 - 相当于遮罩的并集
            merged = torch.max(masks, dim=0, keepdim=True)[0]
        elif merge_mode == "min":
            # 取最小值 - 相当于遮罩的交集
            merged = torch.min(masks, dim=0, keepdim=True)[0]
        elif merge_mode == "average":
            # 取平均值
            merged = torch.mean(masks, dim=0, keepdim=True)
        elif merge_mode == "sum":
            # 求和并裁剪到0-1范围
            merged = torch.sum(masks, dim=0, keepdim=True)
            merged = torch.clamp(merged, 0.0, 1.0)
        else:
            # 默认使用最大值
            merged = torch.max(masks, dim=0, keepdim=True)[0]

        return (merged,)
NODE_CLASS_MAPPINGS = {
    "CachePreviewBridge": CachePreviewBridge,
    "LG_Noise": LG_Noise,
    "LG_LoadImage": LG_LoadImage,
    "LG_LatentBatchToList": LG_LatentBatchToList,
    "LG_SaveImage": LG_SaveImage,
    "LG_LoadImage_V2": LG_LoadImage_V2,
    "LG_InstallDependencies": LG_InstallDependencies,
    "LG_PipManager": LG_PipManager,
    "LG_FloatRange": LG_FloatRange,
    "LG_Counter": LG_Counter,
    "LG_ImageLoaderWithCounter": LG_ImageLoaderWithCounter,
    "LG_MaskHoleFiller": LG_MaskHoleFiller,
    "LG_MaskBatchMerge": LG_MaskBatchMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CachePreviewBridge": "🎈LG_PreviewBridge",
    "LG_Noise": "🎈LG_Noise",
    "LG_LoadImage": "🎈LG_LoadImage",
    "LG_LatentBatchToList": "🎈LG_Latent批次转列表",
    "LG_SaveImage": "🎈LG_SaveImage",
    "LG_LoadImage_V2": "🎈LG_LoadImage_V2",
    "LG_InstallDependencies": "🎈LG_安装依赖",
    "LG_PipManager": "🎈LG_Pip管理器",
    "LG_FloatRange": "🎈LG_浮点数[0-1]",
    "LG_Counter": "🎈LG_计数器",
    "LG_ImageLoaderWithCounter": "🎈LG_图片列表加载器",
    "LG_MaskHoleFiller": "🎈LG_填充空洞",
    "LG_MaskBatchMerge": "🎈LG_合并批次遮罩",
}



