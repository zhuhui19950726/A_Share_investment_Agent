"""
Agent 数据收集与聚合模块

在工作流结束后，从 api_state（由 @agent_endpoint 按 run_id 收集）和
final_state（LangGraph 最终状态）中提取各 Agent 的输出，归一化后生成
EnhancedFinalState，供 summary_report 渲染汇总报告。

设计要点：
- final_state.metadata["agent_reasoning"] 会被每个 Agent 覆盖，不能作为逐 Agent 数据源。
- api_state 按 Agent 独立保存 output_state / reasoning，且 history 按 run_id 隔离，
  因此优先从 api_state 取数，final_state 作为 fallback。
- 所有置信度统一归一化为 0~1 浮点。
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.utils.logging_config import setup_logger

logger = setup_logger("agent_collector")


@dataclass
class AgentSignal:
    """单个分析师信号"""

    agent_name: str
    signal: str = ""
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class AgentOutput:
    """单个 Agent 的标准化输出"""

    agent_name: str
    display_name: str = ""
    signal: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    llm_call_count: int = 0
    error: Optional[str] = None


@dataclass
class DecisionSummary:
    """最终交易决策摘要"""

    action: str = "hold"
    quantity: int = 0
    confidence: float = 0.0
    reasoning: str = ""
    agent_signals: List[AgentSignal] = field(default_factory=list)
    report: str = ""


@dataclass
class EnhancedFinalState:
    """聚合后的最终状态，作为 summary_report 的输入"""

    run_id: str
    ticker: str = ""
    start_date: str = ""
    end_date: str = ""
    decision: DecisionSummary = field(default_factory=DecisionSummary)
    agent_outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    llm_call_count: int = 0
    warnings: List[str] = field(default_factory=list)
    generated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为 JSON 可序列化字典"""
        return {
            "run_id": self.run_id,
            "ticker": self.ticker,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "decision": {
                "action": self.decision.action,
                "quantity": self.decision.quantity,
                "confidence": self.decision.confidence,
                "reasoning": self.decision.reasoning,
                "agent_signals": [
                    {
                        "agent_name": s.agent_name,
                        "signal": s.signal,
                        "confidence": s.confidence,
                        "reasoning": s.reasoning,
                    }
                    for s in self.decision.agent_signals
                ],
                "report": self.decision.report,
            },
            "agent_outputs": {
                name: {
                    "agent_name": out.agent_name,
                    "display_name": out.display_name,
                    "signal": out.signal,
                    "confidence": out.confidence,
                    "reasoning": out.reasoning,
                    "llm_call_count": out.llm_call_count,
                    "error": out.error,
                }
                for name, out in self.agent_outputs.items()
            },
            "llm_call_count": self.llm_call_count,
            "warnings": self.warnings,
            "generated_at": self.generated_at,
        }


class AgentCollector:
    """收集并聚合各 Agent 输出的核心类"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._states: Dict[str, EnhancedFinalState] = {}
        self._current_run_id: Optional[str] = None

    @staticmethod
    def _normalize_confidence(value: Any, warnings: List[str]) -> float:
        """把 '75%' / 0.75 / 75 统一归一化为 0~1 浮点"""
        if value is None or isinstance(value, bool):
            return 0.0
        if isinstance(value, str):
            cleaned = value.strip().replace("%", "")
            try:
                value = float(cleaned)
            except ValueError:
                warnings.append(f"无法解析置信度: {value!r}")
                return 0.0
        if isinstance(value, (int, float)):
            number = float(value)
            if 0.0 <= number <= 1.0:
                return number
            if 0.0 <= number <= 100.0:
                return number / 100.0
            warnings.append(f"置信度超出 0~100 范围: {value!r}")
            return 0.0
        warnings.append(f"置信度类型不支持: {type(value).__name__}")
        return 0.0

    @staticmethod
    def _parse_content(content: Any) -> Optional[Dict[str, Any]]:
        """解析消息 content 为字典（支持纯 JSON 与文本中嵌入的 JSON）"""
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return None
        text = content.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _normalize_agent_name(name: str) -> str:
        """统一 Agent 名称：portfolio_management -> portfolio_management_agent"""
        if not name:
            return name
        return name if name.endswith("_agent") else name + "_agent"

    def _get_api_state(self):
        try:
            from backend.state import api_state

            return api_state
        except ImportError as exc:  # pragma: no cover - 仅在缺失后端时触发
            logger.debug("无法导入 backend.state，跳过 api_state 取数: %s", exc)
            return None

    def _collect_from_api_state(self, run_id: str) -> Dict[str, AgentOutput]:
        """从 api_state 的 history 按 run_id 收集各 Agent 的独立输出快照"""
        outputs: Dict[str, AgentOutput] = {}
        api_state = self._get_api_state()
        if api_state is None:
            return outputs

        with api_state._lock:
            agent_data_snapshot = {
                name: {
                    "history": [dict(entry) for entry in data.get("history", [])],
                    "latest": dict(data.get("latest") or {}),
                }
                for name, data in api_state._agent_data.items()
            }

        for agent_name, data in agent_data_snapshot.items():
            fields: Dict[str, Any] = {}
            for entry in data["history"]:
                if entry.get("run_id") == run_id:
                    for key, value in entry.items():
                        if key not in ("run_id", "timestamp"):
                            fields[key] = value
            if not fields:
                fields = data["latest"]

            output_state = fields.get("output_state") or {}
            if not isinstance(output_state, dict):
                output_state = {}
            metadata = output_state.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}

            messages = output_state.get("messages") or []
            parsed: Optional[Dict[str, Any]] = None
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    parsed = self._parse_content(last_message.get("content"))
            if parsed is None:
                reasoning_from_state = metadata.get("agent_reasoning")
                if isinstance(reasoning_from_state, str):
                    parsed = self._parse_content(reasoning_from_state)

            reasoning = fields.get("reasoning") or metadata.get("agent_reasoning")
            if not isinstance(reasoning, str):
                reasoning = None

            outputs[agent_name] = AgentOutput(
                agent_name=agent_name,
                signal=self._extract_signal(parsed),
                confidence=self._extract_confidence(parsed),
                reasoning=reasoning,
                raw_data=parsed or {},
                llm_call_count=self._count_llm_calls_for_agent(data["history"], run_id),
                error=fields.get("error"),
            )
        return outputs

    def _collect_from_messages(self, final_state: Dict[str, Any]) -> Dict[str, AgentOutput]:
        """从 final_state.messages 按消息 name 匹配提取（api_state 不可用时的 fallback）"""
        outputs: Dict[str, AgentOutput] = {}
        for message in final_state.get("messages", []):
            name = getattr(message, "name", None)
            if not name:
                continue
            agent_name = self._normalize_agent_name(name)
            parsed = self._parse_content(getattr(message, "content", None))
            outputs[agent_name] = AgentOutput(
                agent_name=agent_name,
                signal=self._extract_signal(parsed),
                confidence=self._extract_confidence(parsed),
                reasoning=(parsed or {}).get("reasoning"),
                raw_data=parsed or {},
            )
        return outputs

    @staticmethod
    def _extract_signal(parsed: Optional[Dict[str, Any]]) -> Optional[str]:
        if not parsed:
            return None
        for key in ("signal", "perspective", "trading_action", "action"):
            value = parsed.get(key)
            if isinstance(value, str) and value:
                return value.lower()
        return None

    @staticmethod
    def _extract_confidence(parsed: Optional[Dict[str, Any]]) -> Optional[float]:
        if not parsed:
            return None
        for key in ("confidence", "mixed_confidence_diff", "risk_score"):
            if key in parsed:
                value = parsed[key]
                if isinstance(value, (int, float, str)):
                    return float(value) if isinstance(value, (int, float)) else None
        return None

    @staticmethod
    def _count_llm_calls_for_agent(history: List[Dict[str, Any]], run_id: str) -> int:
        return sum(
            1
            for entry in history
            if entry.get("run_id") == run_id and "llm_request" in entry
        )

    def _count_llm_calls(self, run_id: str) -> int:
        api_state = self._get_api_state()
        if api_state is None:
            return 0
        with api_state._lock:
            return sum(
                self._count_llm_calls_for_agent(data.get("history", []), run_id)
                for data in api_state._agent_data.values()
            )

    def _parse_decision(
        self, outputs: Dict[str, AgentOutput], warnings: List[str]
    ) -> DecisionSummary:
        portfolio = outputs.get("portfolio_management_agent")
        raw = portfolio.raw_data if portfolio else {}
        if not raw:
            warnings.append("未找到 portfolio_management_agent 的决策输出")

        action = str(raw.get("action", "hold")).lower()
        quantity = raw.get("quantity", 0)
        if not isinstance(quantity, int):
            try:
                quantity = int(quantity)
            except (TypeError, ValueError):
                warnings.append(f"交易数量解析失败: {quantity!r}")
                quantity = 0

        signals: List[AgentSignal] = []
        for item in raw.get("agent_signals", []) or []:
            if not isinstance(item, dict):
                continue
            signals.append(
                AgentSignal(
                    agent_name=str(item.get("agent_name") or item.get("agent") or "unknown"),
                    signal=str(item.get("signal", "")).lower(),
                    confidence=self._normalize_confidence(item.get("confidence"), warnings),
                    reasoning=str(item.get("reasoning", "")) or "",
                )
            )

        report = raw.get("分析报告") or raw.get("report") or ""
        return DecisionSummary(
            action=action,
            quantity=quantity,
            confidence=self._normalize_confidence(raw.get("confidence"), warnings),
            reasoning=str(raw.get("reasoning") or ""),
            agent_signals=signals,
            report=str(report),
        )

    def store_final_state(self, final_state: Dict[str, Any]) -> None:
        """采集并聚合最终状态，结果按 run_id 存入内存"""
        metadata = final_state.get("metadata") or {}
        data = final_state.get("data") or {}
        run_id = metadata.get("run_id") or str(uuid4())
        warnings: List[str] = []

        outputs = self._collect_from_api_state(run_id)
        for name, output in self._collect_from_messages(final_state).items():
            if name not in outputs or not outputs[name].raw_data:
                outputs[name] = output

        try:
            from src.utils.structured_terminal import AGENT_MAP

            display_names = {
                name: info.get("name", name) for name, info in AGENT_MAP.items()
            }
        except ImportError:
            display_names = {}
        for name, output in outputs.items():
            output.display_name = display_names.get(name, name)
            output.confidence = self._normalize_confidence(
                output.confidence, warnings
            )

        enhanced = EnhancedFinalState(
            run_id=run_id,
            ticker=str(data.get("ticker", "")),
            start_date=str(data.get("start_date", "")),
            end_date=str(data.get("end_date", "")),
            decision=self._parse_decision(outputs, warnings),
            agent_outputs=outputs,
            llm_call_count=self._count_llm_calls(run_id),
            warnings=warnings,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._states[run_id] = enhanced
            self._current_run_id = run_id
        logger.info("已收集运行 %s 的 Agent 输出（%d 个 Agent）", run_id, len(outputs))

    def get_enhanced_final_state(
        self, run_id: Optional[str] = None
    ) -> EnhancedFinalState:
        """获取聚合后的状态；未指定 run_id 时返回最近一次运行"""
        with self._lock:
            target = run_id or self._current_run_id
            if target and target in self._states:
                return self._states[target]
        if target:
            return EnhancedFinalState(
                run_id=target,
                warnings=[f"未找到运行 {target} 的采集结果"],
            )
        return EnhancedFinalState(
            run_id="unknown",
            warnings=["尚未采集任何运行状态"],
        )

    def reset(self, run_id: Optional[str] = None) -> None:
        """清理采集结果"""
        with self._lock:
            if run_id:
                self._states.pop(run_id, None)
                if self._current_run_id == run_id:
                    self._current_run_id = None
            else:
                self._states.clear()
                self._current_run_id = None


# 全局实例与模块级接口（兼容 src/main.py 的现有调用）
_collector = AgentCollector()


def store_final_state(final_state: Dict[str, Any]) -> None:
    """兼容接口：采集 final_state"""
    _collector.store_final_state(final_state)


def get_enhanced_final_state(run_id: Optional[str] = None) -> EnhancedFinalState:
    """兼容接口：获取聚合后的状态"""
    return _collector.get_enhanced_final_state(run_id)


def get_collector() -> AgentCollector:
    """获取全局采集器实例（便于测试与高级用法）"""
    return _collector
