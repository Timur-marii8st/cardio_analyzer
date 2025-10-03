import React from "react";
import ReactECharts from "echarts-for-react";

type Props = { prob: number };

const RiskGauge: React.FC<Props> = ({ prob }) => {
  const val = Math.max(0, Math.min(1, prob)) * 100;
  const option = {
    title: { text: "Риск гипоксии", left: "center" },
    series: [
      {
        type: "gauge",
        progress: { show: true, width: 12 },
        axisLine: { lineStyle: { width: 12 } },
        axisTick: { show: false },
        splitLine: { length: 10, lineStyle: { width: 2 } },
        axisLabel: { distance: 10 },
        pointer: { length: "60%" },
        detail: { valueAnimation: true, formatter: (value: number) => `${value.toFixed(1)}%`, offsetCenter: ['0%', '90%']},
        data: [{ value: val }]
      }
    ],
    visualMap: {
      show: false,
      min: 0, max: 100,
      inRange: { color: ["#2ecc71", "#f1c40f", "#e74c3c"] }
    }
  };
  return <ReactECharts option={option} style={{ height: 220 }} />;
};

export default RiskGauge;