import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

const MODE_BYPASS = 4;
const MODE_MUTE = 2;
const MODE_ALWAYS = 0;

function recomputeInsideNodesForGroup(group) {
    if (!group.graph) return;
    group._nodes = [];
    const nodeBoundings = {};
    for (const node of group.graph._nodes) {
        nodeBoundings[node.id] = node.getBounding();
    }
    for (const node of group.graph._nodes) {
        const bounding = nodeBoundings[node.id];
        if (bounding && LiteGraph.overlapBounding(group._bounding, bounding)) {
            group._nodes.push(node);
        }
    }
}

class LG_GroupMuterNode extends LGraphNode {
    constructor(title) {
        super(title);
        this.isVirtualNode = true;
        this.serialize_widgets = true;
        this.properties = { mode: "mute", maxOne: false };
        this.groupWidgetMap = new Map();
        this.tempSize = null;
        this.debouncerTempWidth = 0;
        this._lastGroupSignature = "";
    }

    onAdded() {
        setTimeout(() => this.setupGroups(), 10);
    }

    setupGroups() {
        if (!app.graph?._groups) return;
        
        const groups = app.graph._groups.sort((a, b) => a.title.localeCompare(b.title));
        const existingGroups = new Set();
        
        groups.forEach((group, index) => {
            const name = group.title;
            existingGroups.add(name);
            
            let widget = this.groupWidgetMap.get(name) || this.widgets?.find(w => w.name === name);
            
            if (!widget) {
                this.tempSize = this.size ? [...this.size] : [300, 60];
                
                if (!this.inputs?.find(i => i.name === name)) {
                    this.addInput(name, "BOOLEAN");
                }
                const input = this.inputs[this.inputs.length - 1];
                
                const widgetResult = ComfyWidgets?.BOOLEAN(this, name, ["BOOLEAN", { default: true }], app);
                widget = widgetResult?.widget;
                
                if (!widget) return;
                
                // Ensure value is explicitly boolean
                if (widget.value !== true && widget.value !== false) {
                    widget.value = true;
                }
                
                input.widget = widget;
                
                // Ensure config symbols are set for PS_Config compatibility
                widget[Symbol.for("GET_CONFIG")] = () => ["BOOLEAN", { default: true }];
                widget[Symbol.for("CONFIG")] = ["BOOLEAN", { default: true }];
                
                const originalCallback = widget.callback;
                widget.callback = (value) => {
                    // Max One模式：开启一个组时，关闭其他所有组
                    if (this.properties.maxOne && value === true) {
                        for (const [otherName, otherWidget] of this.groupWidgetMap) {
                            if (otherName !== name && otherWidget.value === true) {
                                otherWidget.value = false;
                                this.applyGroupMode(otherName, false);
                                const otherGroup = app.graph?._groups.find(g => g.title === otherName);
                                if (otherGroup) otherGroup.lgtools_isActive = false;
                            }
                        }
                    }
                    
                    this.applyGroupMode(name, value);
                    group.lgtools_isActive = value;
                    originalCallback?.call(widget, value);
                };
                
                this.groupWidgetMap.set(name, widget);
                group.lgtools_isActive = widget.value;
                this.setSize(this.computeSize());
            } else {
                this.groupWidgetMap.set(name, widget);
                const input = this.inputs?.find(i => i.name === name);
                if (input && !input.widget) input.widget = widget;
                
                // Ensure config symbols exist for existing widgets
                if (!widget[Symbol.for("GET_CONFIG")]) {
                    widget[Symbol.for("GET_CONFIG")] = () => ["BOOLEAN", { default: true }];
                    widget[Symbol.for("CONFIG")] = ["BOOLEAN", { default: true }];
                }
                
                if (group.lgtools_isActive != null && widget.value !== group.lgtools_isActive) {
                    widget.value = group.lgtools_isActive;
                }
            }
            
            if (this.widgets?.[index] !== widget) {
                const oldIndex = this.widgets.indexOf(widget);
                if (oldIndex !== -1) {
                    this.widgets.splice(index, 0, this.widgets.splice(oldIndex, 1)[0]);
                }
            }
        });
        
        const toRemove = [...this.groupWidgetMap.keys()].filter(name => !existingGroups.has(name));
        if (toRemove.length) {
            const widgetIndices = [];
            const inputIndices = [];
            
            toRemove.forEach(name => {
                const widget = this.groupWidgetMap.get(name);
                if (widget) widgetIndices.push(this.widgets.indexOf(widget));
                const inputIdx = this.inputs?.findIndex(i => i.name === name);
                if (inputIdx !== -1) inputIndices.push(inputIdx);
            });
            
            widgetIndices.sort((a, b) => b - a).forEach(i => this.widgets.splice(i, 1));
            inputIndices.sort((a, b) => b - a).forEach(i => this.removeInput(i));
            
            toRemove.forEach(name => {
                this.groupWidgetMap.delete(name);
            });
            
            this.setSize(this.computeSize());
        }
        
        this.setDirtyCanvas(true, false);
    }

    applyGroupMode(groupName, enabled) {
        const group = app.graph?._groups.find(g => g.title === groupName);
        if (!group) return;
        
        const targetMode = enabled ? MODE_ALWAYS : (this.properties.mode === "mute" ? MODE_MUTE : MODE_BYPASS);
        recomputeInsideNodesForGroup(group);
        
        (group._nodes || []).forEach(node => {
            if (node.mode !== targetMode) node.mode = targetMode;
        });
        
        app.graph.setDirtyCanvas(true, false);
    }

    computeSize(out) {
        const widgetCount = this.widgets?.length || 0;
        let size = [200, (LiteGraph.NODE_TITLE_HEIGHT || 30) + widgetCount * (LiteGraph.NODE_WIDGET_HEIGHT || 20)];
        
        if (this.tempSize) {
            size = [Math.max(this.tempSize[0], size[0]), Math.max(this.tempSize[1], size[1])];
            clearTimeout(this.debouncerTempWidth);
            this.debouncerTempWidth = setTimeout(() => this.tempSize = null, 32);
        }
        
        if (out) {
            out[0] = size[0];
            out[1] = size[1];
        }
        return size;
    }

    getExtraMenuOptions(canvas, options) {
        const currentMode = this.properties.mode || "mute";
        const nextMode = currentMode === "mute" ? "bypass" : "mute";
        
        options.push({
            content: `Switch to ${nextMode === "mute" ? "Mute" : "Bypass"} mode`,
            callback: () => {
                this.properties.mode = nextMode;
                for (const [name, widget] of this.groupWidgetMap) {
                    this.applyGroupMode(name, widget.value);
                }
            }
        });
        
        options.push({
            content: this.properties.maxOne ? "✓ Max One Mode" : "Max One Mode",
            callback: () => {
                this.properties.maxOne = !this.properties.maxOne;
                // 如果启用Max One模式，确保最多只有一个组是开启的
                if (this.properties.maxOne) {
                    let firstActiveFound = false;
                    for (const [name, widget] of this.groupWidgetMap) {
                        if (widget.value === true) {
                            if (firstActiveFound) {
                                widget.value = false;
                                this.applyGroupMode(name, false);
                                const group = app.graph?._groups.find(g => g.title === name);
                                if (group) group.lgtools_isActive = false;
                            } else {
                                firstActiveFound = true;
                            }
                        }
                    }
                }
                this.setDirtyCanvas(true, false);
            }
        });
        
        options.push({
            content: "Toggle All Groups",
            callback: () => {
                const newValue = !(this.widgets || []).every(w => w.value);
                (this.widgets || []).forEach(w => {
                    w.value = newValue;
                    w.callback?.(newValue);
                });
            }
        });
    }

    onDrawBackground() {
        if (!app.graph?._groups) return;
        
        const signature = app.graph._groups.map(g => g.title).sort().join(',');
        if (signature !== this._lastGroupSignature) {
            this._lastGroupSignature = signature;
            this.setupGroups();
        }
        
        for (const [name, widget] of this.groupWidgetMap) {
            const group = app.graph._groups.find(g => g.title === name);
            if (!group) continue;
            
            recomputeInsideNodesForGroup(group);
            if (!group._nodes?.length) continue;
            
            const isActive = group._nodes.some(n => n.mode === MODE_ALWAYS);
            group.lgtools_isActive = isActive;
            
            if (widget.value !== isActive) {
                widget.value = isActive;
                this.setDirtyCanvas(true, false);
            }
        }
    }

    static setUp() {
        LiteGraph.registerNodeType(this.type, this);
        if (this._category) {
            this.category = this._category;
        }
    }
}

LG_GroupMuterNode.type = "🎈LG_GroupMuter";
LG_GroupMuterNode.title = "🎈LG Group Muter";
LG_GroupMuterNode._category = "🎈LAOGOU/Switch";
LG_GroupMuterNode["@mode"] = { type: "combo", values: ["mute", "bypass"] };

app.registerExtension({
    name: "LG.GroupMuter",
    registerCustomNodes() {
        LG_GroupMuterNode.setUp();
    },
    loadedGraphNode(node) {
        if (node.type === LG_GroupMuterNode.type) {
            node.tempSize = node.size ? [...node.size] : null;
        }
    }
});
