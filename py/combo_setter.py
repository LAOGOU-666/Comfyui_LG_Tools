CATEGORY_TYPE = "🎈LAOGOU/Utils"

class ComboSetter:
    """
    动态Combo设置器
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "labels": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "每行一个标签"
                }),
                "prompts": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "每行一个提示词"
                }),
                "selected": ("STRING", {
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("selected_label", "selected_prompt",)
    FUNCTION = "execute"
    CATEGORY = CATEGORY_TYPE

    def execute(self, labels, prompts, selected):
        # 按行分割
        label_lines = [line.strip() for line in labels.split('\n') if line.strip()]
        prompt_lines = [line.strip() for line in prompts.split('\n') if line.strip()]
        
        # 找到选中的索引
        selected_prompt = ""
        if selected in label_lines:
            index = label_lines.index(selected)
            if index < len(prompt_lines):
                selected_prompt = prompt_lines[index]
        
        return (selected, selected_prompt)

NODE_CLASS_MAPPINGS = {
    "ComboSetter": ComboSetter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComboSetter": "🎈ComboSetter",
}

