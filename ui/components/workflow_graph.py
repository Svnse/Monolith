"""Read-only workflow graph preview used by the Workshop."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from core import style as _s

_MARGIN_X = 12.0
_MARGIN_Y = 14.0
_NODE_MIN_W = 72.0
_NODE_MAX_W = 112.0
_NODE_H = 44.0
_GAP_X = 10.0
_GAP_Y = 12.0
_MIN_H = 148


@dataclass(frozen=True)
class _GraphNode:
    id: str
    label: str
    kind: str
    x: float
    y: float


@dataclass(frozen=True)
class _GraphWire:
    source: str
    target: str


class WorkflowGraphView(QWidget):
    """Compact native preview of a Monoline workflow."""

    sig_node_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflow_graph_view")
        self.setMinimumHeight(_MIN_H)
        policy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self._nodes: list[_GraphNode] = []
        self._wires: list[_GraphWire] = []
        self._rects: dict[str, QRectF] = {}
        self._selected_node_id = ""
        self._run_model: Any | None = None
        self._title = "No workflow"
        self.setStyleSheet(
            f"""
            #workflow_graph_view {{
                background: {_s.BG_PANEL};
                border: 1px solid {_s.BORDER_SUBTLE};
                border-radius: 6px;
            }}
            """
        )

    def bind_workflow(self, workflow: Any | None) -> None:
        self._run_model = None
        self._selected_node_id = ""
        self._load_workflow(workflow)
        self._refresh_height()
        self.update()

    def bind_run_model(self, model: Any | None) -> None:
        self._run_model = model
        self.update()

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._height_for_width(float(width))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(360, self.heightForWidth(360))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(220, self.heightForWidth(220))

    def resizeEvent(self, event: object) -> None:  # noqa: N802
        self._refresh_height()
        super().resizeEvent(event)

    def node_count(self) -> int:
        return len(self._nodes)

    def wire_count(self) -> int:
        return len(self._wires)

    def selected_node_id(self) -> str:
        return self._selected_node_id

    def node_labels(self) -> list[str]:
        return [node.label for node in self._nodes]

    def status_for(self, block_id: str) -> str:
        return self._block_status(block_id)

    def layout_row_count_for_width(self, width: int) -> int:
        return self._row_count_for_width(float(width))

    def _load_workflow(self, workflow: Any | None) -> None:
        self._nodes = []
        self._wires = []
        if workflow is None:
            self._title = "No workflow"
            return

        self._title = str(getattr(workflow, "name", "") or "Workflow")
        source_path = getattr(workflow, "source_path", None)
        if not source_path:
            self._nodes = [_GraphNode(str(getattr(workflow, "id", "native")), self._title, "native", 0.0, 0.0)]
            return

        try:
            data = json.loads(Path(source_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._nodes = [_GraphNode(str(getattr(workflow, "id", "missing")), self._title, "missing", 0.0, 0.0)]
            return

        blocks = data.get("blocks") if isinstance(data, dict) else []
        if not isinstance(blocks, list):
            blocks = []
        self._nodes = [self._node_from_block(block, idx) for idx, block in enumerate(blocks) if isinstance(block, dict)]
        self._wires = self._parse_wires(data.get("connections") if isinstance(data, dict) else [])

    def _node_from_block(self, block: dict[str, Any], idx: int) -> _GraphNode:
        block_id = str(block.get("id") or f"block-{idx + 1}")
        kind = str(block.get("kind") or "block")
        config = block.get("config") if isinstance(block.get("config"), dict) else {}
        label = str(block.get("label") or config.get("label") or block_id)
        position = block.get("position")
        if not (isinstance(position, list) and len(position) >= 2):
            position = [idx * 220.0, 0.0]
        try:
            x = float(position[0])
            y = float(position[1])
        except (TypeError, ValueError):
            x = float(idx * 220)
            y = 0.0
        return _GraphNode(block_id, label, kind, x, y)

    def _parse_wires(self, connections: Any) -> list[_GraphWire]:
        if not isinstance(connections, list):
            return []
        known_ids = {node.id for node in self._nodes}
        wires: list[_GraphWire] = []
        for connection in connections:
            source = ""
            target = ""
            if isinstance(connection, dict):
                source = str(connection.get("from_block") or connection.get("source") or "").split(".", 1)[0]
                target = str(connection.get("to_block") or connection.get("target") or "").split(".", 1)[0]
            elif isinstance(connection, list) and len(connection) >= 2:
                source = str(connection[0]).split(".", 1)[0]
                target = str(connection[1]).split(".", 1)[0]
            if source in known_ids and target in known_ids:
                wires.append(_GraphWire(source, target))
        return wires

    def _layout_rects(self) -> dict[str, QRectF]:
        rect = QRectF(self.rect()).adjusted(_MARGIN_X, _MARGIN_Y, -_MARGIN_X, -_MARGIN_Y)
        if not self._nodes or rect.width() <= 0 or rect.height() <= 0:
            return {}

        if self._should_flow_layout(rect.width()):
            return self._flow_rects(rect)

        node_w = max(_NODE_MIN_W, min(_NODE_MAX_W, (rect.width() - _GAP_X * max(0, len(self._nodes) - 1)) / max(1, len(self._nodes))))
        usable_w = max(1.0, rect.width() - node_w)
        usable_h = max(1.0, rect.height() - _NODE_H)
        xs = [node.x for node in self._nodes]
        ys = [node.y for node in self._nodes]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)
        ordered = sorted(self._nodes, key=lambda item: (item.x, item.y, item.id))
        fallback_index = {node.id: idx for idx, node in enumerate(ordered)}
        rects: dict[str, QRectF] = {}
        for node in self._nodes:
            if dx < 1 or dy < 1:
                x = rect.left() + usable_w * (fallback_index[node.id] / max(1, len(self._nodes) - 1))
            else:
                x = rect.left() + usable_w * ((node.x - min(xs)) / dx)
            if dy < 1:
                y = rect.center().y() - _NODE_H / 2
            else:
                y = rect.top() + usable_h * ((node.y - min(ys)) / dy)
            rects[node.id] = QRectF(x, y, node_w, _NODE_H)
        return rects

    def _flow_rects(self, rect: QRectF) -> dict[str, QRectF]:
        columns = self._columns_for_inner_width(rect.width())
        node_w = max(_NODE_MIN_W, min(_NODE_MAX_W, (rect.width() - _GAP_X * max(0, columns - 1)) / max(1, columns)))
        ordered = sorted(self._nodes, key=lambda item: (item.x, item.y, item.id))
        rows = max(1, (len(ordered) + columns - 1) // columns)
        total_h = rows * _NODE_H + max(0, rows - 1) * _GAP_Y
        top = rect.top() + max(0.0, (rect.height() - total_h) / 2)
        rects: dict[str, QRectF] = {}
        for idx, node in enumerate(ordered):
            row = idx // columns
            col = idx % columns
            x = rect.left() + col * (node_w + _GAP_X)
            y = top + row * (_NODE_H + _GAP_Y)
            rects[node.id] = QRectF(x, y, node_w, _NODE_H)
        return rects

    def _refresh_height(self) -> None:
        target = self.heightForWidth(max(1, self.width()))
        if target != self.minimumHeight():
            self.setMinimumHeight(target)
            self.updateGeometry()

    def _height_for_width(self, width: float) -> int:
        rows = self._row_count_for_width(width)
        content = rows * _NODE_H + max(0, rows - 1) * _GAP_Y
        return int(max(_MIN_H, content + _MARGIN_Y * 2))

    def _row_count_for_width(self, width: float) -> int:
        if not self._nodes:
            return 1
        inner_w = max(1.0, width - _MARGIN_X * 2)
        if not self._should_flow_layout(inner_w):
            return 1
        columns = self._columns_for_inner_width(inner_w)
        return max(1, (len(self._nodes) + columns - 1) // columns)

    def _columns_for_inner_width(self, inner_w: float) -> int:
        return max(1, int((inner_w + _GAP_X) // (_NODE_MIN_W + _GAP_X)))

    def _should_flow_layout(self, inner_w: float) -> bool:
        if len(self._nodes) <= 1:
            return False
        required_w = len(self._nodes) * _NODE_MIN_W + max(0, len(self._nodes) - 1) * _GAP_X
        return required_w > inner_w

    def paintEvent(self, event: object) -> None:  # noqa: D401
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._rects = self._layout_rects()
        if not self._nodes:
            self._draw_empty(painter)
            painter.end()
            return
        self._draw_wires(painter)
        for node in self._nodes:
            self._draw_node(painter, node, self._rects.get(node.id))
        painter.end()

    def mouseReleaseEvent(self, event: object) -> None:
        pos = event.position().toPoint() if hasattr(event, "position") else None
        if pos is None:
            return
        for node_id, rect in self._rects.items():
            if rect.contains(QPointF(pos)):
                self._selected_node_id = node_id
                self.sig_node_selected.emit(node_id)
                self.update()
                return

    def _draw_empty(self, painter: QPainter) -> None:
        painter.setPen(QColor(_s.FG_DIM))
        painter.drawText(QRectF(self.rect()).adjusted(10, 0, -10, 0), Qt.AlignmentFlag.AlignCenter, self._title)

    def _draw_wires(self, painter: QPainter) -> None:
        pen = QPen(QColor(_s.BORDER_LIGHT), 1.4)
        painter.setPen(pen)
        for wire in self._wires:
            src = self._rects.get(wire.source)
            dst = self._rects.get(wire.target)
            if src is None or dst is None:
                continue
            start = QPointF(src.right(), src.center().y())
            end = QPointF(dst.left(), dst.center().y())
            painter.drawLine(start, end)

    def _draw_node(self, painter: QPainter, node: _GraphNode, rect: QRectF | None) -> None:
        if rect is None:
            return
        status = self._block_status(node.id)
        border = QColor(self._status_color(status))
        fill = QColor(_s.BG_SURFACE_2 if status != "running" else "#192333")
        if node.id == self._selected_node_id:
            border = QColor(_s.ACCENT_PRIMARY)
            fill = QColor("#1d2630")
        painter.setPen(QPen(border, 1.4))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 5, 5)

        stripe = QRectF(rect.left(), rect.top(), 4, rect.height())
        painter.fillRect(stripe, border)

        text_rect = rect.adjusted(9, 5, -7, -5)
        metrics = QFontMetrics(painter.font())
        painter.setPen(QColor(_s.FG_TEXT))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            metrics.elidedText(node.label, Qt.TextElideMode.ElideRight, int(text_rect.width())),
        )
        painter.setPen(QColor(_s.FG_DIM))
        painter.drawText(
            text_rect.adjusted(0, 18, 0, 0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            metrics.elidedText(node.kind, Qt.TextElideMode.ElideRight, int(text_rect.width())),
        )

    def _block_status(self, block_id: str) -> str:
        model = self._run_model
        if model is None or not hasattr(model, "block"):
            return ""
        block = model.block(block_id)
        return str(getattr(block, "status", "") or "") if block is not None else ""

    def _status_color(self, status: str) -> str:
        if status == "done":
            return "#66b87a"
        if status == "running":
            return _s.ACCENT_PRIMARY
        if status == "error":
            return _s.FG_ERROR
        if status == "pending":
            return _s.FG_DIM
        return _s.BORDER_LIGHT
