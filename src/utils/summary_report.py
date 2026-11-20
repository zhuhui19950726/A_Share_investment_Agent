"""
汇总报告渲染模块

将 agent_collector.EnhancedFinalState 渲染为美观、紧凑的终端汇总报告。
与 structured_terminal.py（逐 Agent 详细推理输出）互补：
本模块聚焦于"最终决策 + 关键信号 + 风险"的一页式摘要。
"""

from typing import Any, Dict, List

from src.utils.agent_collector import AgentSignal, EnhancedFinalState
from src.utils.structured_terminal import AGENT_MAP, AGENT_ORDER, STATUS_ICONS, SYMBOLS


class SummaryReportRenderer:
    """把 EnhancedFinalState 渲染为终端文本"""

    def __init__(self, width: int = 80) -> None:
        self.width = width

    def render(self, state: EnhancedFinalState) -> str:
        """渲染完整汇总报告文本"""
        lines: List[str] = []
        lines.extend(self._render_header(state))
        lines.extend(self._render_decision(state))
        lines.extend(self._render_signals(state))
        lines.extend(self._render_risk_and_debate(state))
        lines.extend(self._render_warnings(state))
        lines.append(SYMBOLS["border"] * self.width)
        return "\n".join(lines)

    def _title_line(self, text: str) -> str:
        title = f" {text} "
        remaining = max(self.width - len(title) - 2, 0)
        left = remaining // 2
        right = remaining - left
        return (
            f"{SYMBOLS['header_left']}{SYMBOLS['border'] * left}{title}"
            f"{SYMBOLS['border'] * right}{SYMBOLS['header_right']}"
        )

    def _field(self, label: str, value: Any) -> str:
        return f"{SYMBOLS['vertical']} {label}: {value}"

    def _wrap_text(self, text: str, indent: str = "   ") -> List[str]:
        """把长文本按报告宽度折行"""
        if not text:
            return []
        max_len = self.width - len(indent) - 2
        result: List[str] = []
        for paragraph in str(text).splitlines():
            if not paragraph.strip():
                continue
            for i in range(0, len(paragraph), max_len):
                result.append(f"{SYMBOLS['vertical']}{indent}{paragraph[i : i + max_len]}")
        return result

    def _render_header(self, state: EnhancedFinalState) -> List[str]:
        lines: List[str] = [
            SYMBOLS["border"] * self.width,
            f"股票 {state.ticker or '未知'} 投资分析汇总报告".center(self.width),
            SYMBOLS["border"] * self.width,
        ]
        if state.start_date or state.end_date:
            lines.append(
                self._field(
                    "分析区间",
                    f"{state.start_date or 'N/A'} 至 {state.end_date or 'N/A'}",
                )
            )
        lines.append(self._field("运行 ID", state.run_id))
        if state.generated_at:
            lines.append(self._field("生成时间", state.generated_at))
        lines.append("")
        return lines

    def _render_decision(self, state: EnhancedFinalState) -> List[str]:
        decision = state.decision
        lines: List[str] = [self._title_line("最终决策")]
        action_icon = STATUS_ICONS.get(decision.action.lower(), "")
        lines.append(
            self._field(
                "操作建议",
                f"{action_icon} {decision.action.upper() if decision.action else 'N/A'}",
            )
        )
        lines.append(self._field("交易数量", f"{decision.quantity} 股"))
        lines.append(
            self._field(
                "决策信心",
                f"{decision.confidence:.0%}" if decision.confidence else "N/A",
            )
        )
        if decision.reasoning:
            lines.append(self._field("决策理由", ""))
            lines.extend(self._wrap_text(decision.reasoning))
        lines.append("")
        return lines

    def _render_signals(self, state: EnhancedFinalState) -> List[str]:
        signals: List[AgentSignal] = list(state.decision.agent_signals)
        if not signals:
            for name in AGENT_ORDER:
                output = state.agent_outputs.get(name)
                if output and output.signal:
                    signals.append(
                        AgentSignal(
                            agent_name=output.display_name or name,
                            signal=output.signal,
                            confidence=output.confidence or 0.0,
                            reasoning=output.reasoning or "",
                        )
                    )
        if not signals:
            return []

        lines: List[str] = [self._title_line("关键信号")]
        for signal in signals:
            icon = STATUS_ICONS.get(signal.signal.lower(), "")
            lines.append(
                self._field(
                    signal.agent_name,
                    f"{icon} {signal.signal} (置信度: {signal.confidence:.0%})",
                )
            )
        lines.append("")
        return lines

    def _render_risk_and_debate(self, state: EnhancedFinalState) -> List[str]:
        lines: List[str] = []
        risk = state.agent_outputs.get("risk_management_agent")
        if risk and risk.raw_data:
            lines.append(self._title_line("风险评估"))
            lines.append(self._field("风险评分", f"{risk.raw_data.get('risk_score', 'N/A')}/10"))
            lines.append(self._field("建议动作", risk.raw_data.get("trading_action", "N/A")))
            metrics = risk.raw_data.get("risk_metrics") or {}
            if isinstance(metrics, dict) and metrics:
                for key, value in metrics.items():
                    if isinstance(value, float) and 0 <= value <= 1:
                        value = f"{value:.1%}"
                    lines.append(self._field(key, value))
            lines.append("")

        debate = state.agent_outputs.get("debate_room_agent")
        if debate and debate.raw_data:
            lines.append(self._title_line("辩论室结论"))
            lines.append(self._field("信号", debate.raw_data.get("signal", "N/A")))
            lines.append(self._field("置信度", f"{debate.confidence:.0%}" if debate.confidence else "N/A"))
            summary = debate.raw_data.get("debate_summary") or debate.raw_data.get("reasoning")
            if summary:
                lines.extend(self._wrap_text(str(summary)))
            lines.append("")

        stats: List[str] = []
        if state.llm_call_count:
            stats.append(f"LLM 调用 {state.llm_call_count} 次")
        if state.agent_outputs:
            stats.append(f"参与 Agent {len(state.agent_outputs)} 个")
        if stats:
            lines.append(self._field("统计", " / ".join(stats)))
            lines.append("")
        return lines

    def _render_warnings(self, state: EnhancedFinalState) -> List[str]:
        if not state.warnings:
            return []
        lines: List[str] = [self._title_line("提示")]
        for warning in state.warnings:
            lines.append(f"{SYMBOLS['vertical']} ⚠️ {warning}")
        lines.append("")
        return lines


def render_summary_report(state: EnhancedFinalState, width: int = 80) -> str:
    """渲染汇总报告文本（不输出）"""
    return SummaryReportRenderer(width=width).render(state)


def print_summary_report(state: EnhancedFinalState) -> None:
    """采集结果的终端输出入口，兼容 src/main.py 的现有调用"""
    print("\n" + render_summary_report(state))
