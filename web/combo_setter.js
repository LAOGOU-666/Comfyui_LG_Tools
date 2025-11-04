import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "ComboSetter",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ComboSetter") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 获取widgets
                const labelsWidget = this.widgets?.find(w => w.name === "labels");
                const promptsWidget = this.widgets?.find(w => w.name === "prompts");
                let selectedWidget = this.widgets?.find(w => w.name === "selected");
                
                if (!labelsWidget || !promptsWidget || !selectedWidget) {
                    console.error("ComboSetter: 无法找到所需的widgets");
                    return result;
                }
                
                // 将selected widget转换为combo类型
                const selectedIndex = this.widgets.indexOf(selectedWidget);
                if (selectedWidget.type !== "combo") {
                    this.widgets.splice(selectedIndex, 1);
                    selectedWidget = this.addWidget("combo", "selected", "", () => {}, {
                        values: [""]
                    });
                    // 将新的combo widget移动到正确的位置
                    const newWidget = this.widgets.pop();
                    this.widgets.splice(selectedIndex, 0, newWidget);
                    selectedWidget = newWidget;
                }
                
                // 添加"Set Combo"按钮
                const setComboBtn = this.addWidget("button", "Set Combo", null, () => {
                    updateComboOptions.call(this);
                });
                
                // 设置按钮样式
                setComboBtn.serialize = false;
                
                // 更新Combo选项的函数
                const updateComboOptions = function() {
                    const labelsText = labelsWidget.value || "";
                    const promptsText = promptsWidget.value || "";
                    
                    // 按行分割labels
                    const labelLines = labelsText.split('\n')
                        .map(line => line.trim())
                        .filter(line => line.length > 0);
                    
                    if (labelLines.length === 0) {
                        console.warn("ComboSetter: labels为空");
                        selectedWidget.options.values = [""];
                        selectedWidget.value = "";
                        return;
                    }
                    
                    // 更新combo的选项
                    selectedWidget.options.values = labelLines;
                    
                    // 如果当前选中的值不在新的选项中，设置为第一个选项
                    if (!labelLines.includes(selectedWidget.value)) {
                        selectedWidget.value = labelLines[0];
                    }
                    
                    // 触发更新
                    this.setDirtyCanvas(true, false);
                    
                    console.log("ComboSetter: 已更新Combo选项", labelLines);
                };
                
                // 自动计算节点大小
                const originalComputeSize = this.computeSize;
                this.computeSize = function(out) {
                    let size = originalComputeSize ? originalComputeSize.apply(this, arguments) : [200, 100];
                    
                    // 根据widgets数量动态调整高度
                    const widgetHeight = 40;
                    const buttonHeight = 30;
                    const padding = 20;
                    
                    let totalHeight = padding;
                    
                    // 计算多行文本框的高度
                    if (labelsWidget) {
                        const lines = (labelsWidget.value || "").split('\n').length;
                        totalHeight += Math.max(lines * 20, 60);
                    }
                    
                    if (promptsWidget) {
                        const lines = (promptsWidget.value || "").split('\n').length;
                        totalHeight += Math.max(lines * 20, 60);
                    }
                    
                    // 添加combo和按钮的高度
                    totalHeight += widgetHeight + buttonHeight + padding;
                    
                    size[1] = totalHeight;
                    size[0] = Math.max(size[0], 300);
                    
                    return size;
                };
                
                return result;
            };
            
            // 序列化时保存combo的值
            const onSerialize = nodeType.prototype.onSerialize;
            nodeType.prototype.onSerialize = function(o) {
                const result = onSerialize?.apply(this, arguments);
                
                const selectedWidget = this.widgets?.find(w => w.name === "selected");
                if (selectedWidget && selectedWidget.options && selectedWidget.options.values) {
                    o.selected_options = selectedWidget.options.values;
                }
                
                return result;
            };
            
            // 反序列化时恢复combo的选项
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function(o) {
                const result = onConfigure?.apply(this, arguments);
                
                if (o.selected_options) {
                    const selectedWidget = this.widgets?.find(w => w.name === "selected");
                    if (selectedWidget) {
                        selectedWidget.options = selectedWidget.options || {};
                        selectedWidget.options.values = o.selected_options;
                    }
                }
                
                return result;
            };
        }
    }
});

