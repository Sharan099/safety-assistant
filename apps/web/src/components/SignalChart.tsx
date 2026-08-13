"use client";

import ReactECharts from "echarts-for-react";

// UI_UX_DESIGN_BRIEF.md Section 13: Signal Workspace time-history chart.
// Overlay Run A / Run B, mark the divergence event if one was found.
export function SignalChart({
  timeS,
  runA,
  runB,
  unit,
  divergenceMs,
}: {
  timeS: number[];
  runA: number[];
  runB: number[];
  unit: string | null;
  divergenceMs?: number | null;
}) {
  const timeMs = timeS.map((t) => Math.round(t * 1000 * 100) / 100);

  const option = {
    backgroundColor: "transparent",
    grid: { left: 56, right: 24, top: 32, bottom: 40 },
    legend: { data: ["Run A", "Run B"], top: 0, textStyle: { color: "#d4d4d4" } },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      name: "time (ms)",
      nameLocation: "middle",
      nameGap: 28,
      data: timeMs,
      axisLabel: { color: "#a3a3a3" },
      axisLine: { lineStyle: { color: "#525252" } },
    },
    yAxis: {
      type: "value",
      name: unit ?? "",
      axisLabel: { color: "#a3a3a3" },
      axisLine: { lineStyle: { color: "#525252" } },
      splitLine: { lineStyle: { color: "#262626" } },
    },
    series: [
      {
        name: "Run A",
        type: "line",
        data: runA,
        showSymbol: false,
        lineStyle: { color: "#38bdf8", width: 1.5 },
        markLine:
          divergenceMs != null
            ? {
                symbol: "none",
                label: { formatter: `divergence: ${divergenceMs.toFixed(1)} ms`, color: "#f59e0b" },
                lineStyle: { color: "#f59e0b", type: "dashed" },
                data: [{ xAxis: timeMs.findIndex((t) => t >= divergenceMs) }],
              }
            : undefined,
      },
      {
        name: "Run B",
        type: "line",
        data: runB,
        showSymbol: false,
        lineStyle: { color: "#f472b6", width: 1.5 },
      },
    ],
  };

  return <ReactECharts option={option} style={{ height: 320, width: "100%" }} theme="dark" />;
}
