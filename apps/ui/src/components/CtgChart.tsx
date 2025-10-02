import React from "react";
import ReactECharts from "echarts-for-react";

type Props = {
  ts: string[];
  bpm: number[];
  baseline: number[];
  ua: number[];
  decelEvents: { start_ts: string; end_ts: string }[];
};

const CtgChart: React.FC<Props> = ({ ts, bpm, baseline, ua, decelEvents }) => {
  const markAreas = decelEvents.map(ev => ({
    name: "Decel",
    itemStyle: { color: "rgba(255, 0, 0, 0.08)" },
    label: { show: false },
    xAxis: ev.start_ts, xAxis2: ev.end_ts
  }));

  const option = {
    title: { text: "CTG", left: "center" },
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    legend: { data: ["FHR", "Baseline", "UA"], top: 24 },
    grid: { left: 50, right: 50, top: 60, bottom: 40 },
    xAxis: {
      type: "category",
      data: ts,
      axisLabel: { formatter: (v: string) => v.slice(11, 19) } // HH:MM:SS
    },
    yAxis: [
      {
        type: "value",
        name: "FHR (bpm)",
        min: 50, max: 220, position: "left"
      },
      {
        type: "value",
        name: "UA",
        min: 0, max: 120, position: "right"
      }
    ],
    series: [
      {
        name: "FHR",
        type: "line",
        yAxisIndex: 0,
        data: bpm,
        showSymbol: false,
        lineStyle: { color: "#1f77b4", width: 2 },
        markArea: { data: markAreas.length ? markAreas.map(m => [{ xAxis: m.xAxis }, { xAxis: m.xAxis2 }]) : [] }
      },
      {
        name: "Baseline",
        type: "line",
        yAxisIndex: 0,
        data: baseline,
        showSymbol: false,
        lineStyle: { color: "#2ca02c", width: 1.5, type: "dashed" }
      },
      {
        name: "UA",
        type: "line",
        yAxisIndex: 1,
        data: ua,
        showSymbol: false,
        areaStyle: { color: "rgba(255,165,0,0.2)" },
        lineStyle: { color: "#ff7f0e", width: 1.5 }
      }
    ]
  };

  return <ReactECharts option={option} style={{ height: 420, width: "100%" }} />;
};

export default CtgChart;