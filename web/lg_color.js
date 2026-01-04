/**
 * File: lg_color_widget.js
 * LG Color Picker Widget for ComfyUI
 */

import { app } from "../../scripts/app.js";

const LGWidgets = {
  LGCOLOR: (key, val, compute = false) => {
    const widget = {};
    widget.y = 0;
    widget.name = key;
    widget.type = "LGCOLOR";
    widget.options = { default: "#808080" };
    widget.value = val || "#808080";

    widget.draw = function (ctx, node, widgetWidth, widgetY, height) {
      const hide = this.type !== "LGCOLOR" && app.canvas.ds.scale > 0.5;
      if (hide) {
        return;
      }

      const border = 3;
      const H = height || 32;

      // Draw background
      ctx.fillStyle = "#000";
      ctx.fillRect(0, widgetY, widgetWidth, H);

      // Draw color preview
      ctx.fillStyle = this.value;
      ctx.fillRect(
        border,
        widgetY + border,
        widgetWidth - border * 2,
        H - border * 2
      );

      // Calculate brightness to determine text color
      const color = hexToRgb(this.value);
      if (color) {
        const brightness = (color.r * 299 + color.g * 587 + color.b * 114) / 1000;
        ctx.fillStyle = brightness > 125 ? "#000" : "#fff";
      } else {
        ctx.fillStyle = "#fff";
      }

      ctx.font = "14px Arial";
      ctx.textAlign = "center";
      ctx.fillText(this.name, widgetWidth * 0.5, widgetY + H / 2 + 5);
    };

    widget.mouse = function (e, pos, node) {
      if (e.type === "pointerdown") {
        const widgets = node.widgets.filter((w) => w.type === "LGCOLOR");

        for (const w of widgets) {
          // Check if click is within this widget's bounds
          const rect = [w.last_y, w.last_y + 32];
          if (pos[1] > rect[0] && pos[1] < rect[1]) {
            // Create color picker input
            const picker = document.createElement("input");
            picker.type = "color";
            picker.value = this.value;

            // Position off-screen
            picker.style.position = "absolute";
            picker.style.left = "-9999px";
            picker.style.top = "-9999px";

            document.body.appendChild(picker);

            picker.addEventListener("change", () => {
              this.value = picker.value;
              node.graph._version++;
              node.setDirtyCanvas(true, true);
              picker.remove();
            });

            picker.addEventListener("blur", () => {
              picker.remove();
            });

            picker.click();
          }
        }
      }
    };

    widget.computeSize = function (width) {
      return [width, 32];
    };

    return widget;
  },
};

// Helper function to convert hex to RGB
function hexToRgb(hex) {
  if (!hex) return null;
  hex = hex.replace(/^#/, "");
  if (hex.length === 3) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  }
  const result = /^([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

// Register extension
const lg_color_widget = {
  name: "LG.ColorWidget",

  init: async () => {
    console.log("Registering LG.ColorWidget");
  },

  getCustomWidgets: function () {
    return {
      LGCOLOR: (node, inputName, inputData, app) => {
        console.debug("Registering LGCOLOR widget, inputData:", inputData);
        // inputData 是一个数组，第一个元素是类型，第二个元素是选项对象
        const defaultValue = inputData?.[1]?.default || "#808080";
        return {
          widget: node.addCustomWidget(
            LGWidgets.LGCOLOR(inputName, defaultValue)
          ),
          minWidth: 150,
          minHeight: 30,
        };
      },
    };
  },
};

app.registerExtension(lg_color_widget);

export { LGWidgets };

