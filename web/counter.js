import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "LG.Counter",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "LG_Counter") {
            
            // 保存原始的 onNodeCreated 方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // 添加刷新按钮
                this.addWidget("button", "refresh", "刷新计数器", () => {
                    this.resetCounter();
                });
                
                return r;
            };
            
            // 添加重置计数器的方法
            nodeType.prototype.resetCounter = async function() {
                try {
                    const response = await api.fetchApi("/counter/reset", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify({
                            node_id: this.id.toString()
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === "success") {
                        console.log("计数器已重置:", result.message);
                    } else {
                        console.error("重置计数器失败:", result.message);
                    }
                } catch (error) {
                    console.error("重置计数器时发生错误:", error);
                }
            };
            
            // 添加右键菜单选项
            const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                const r = getExtraMenuOptions ? getExtraMenuOptions.apply(this, arguments) : undefined;
                
                options.unshift({
                    content: "重置计数器",
                    callback: () => {
                        this.resetCounter();
                    }
                });
                
                return r;
            };
        }
    }
});

