"""
graph.py — LangGraph Prediction Graph
──────────────────────────────────────

orchestrator_node
      │
      ▼
price_node
      │
      ├──────────────────────┬──────────────────────┐
      ▼                      ▼                      ▼
technical_node          hints_node           oi_node (F&O only)
      │                      │                      │
      └──────────────┬────────┴──────────────────────┘
                     ▼
            correlation_node
                     │
                     ▼
              pattern_node
                     │
                     ▼
            prediction_node
                     │
                     ▼
                    END
"""

from langgraph.graph import StateGraph, END
from state import PredictionState
from nodes.orchestrator_node import orchestrator_node
from nodes.price_node        import price_node
from nodes.technical_node    import technical_node
from nodes.oi_node           import oi_node
from nodes.hints_node        import hints_node
from nodes.correlation_node  import correlation_node
from nodes.pattern_node      import pattern_node
from nodes.prediction_node   import prediction_node


def build_graph():
    g = StateGraph(PredictionState)

    g.add_node("orchestrator_node", orchestrator_node)
    g.add_node("price_node",        price_node)
    g.add_node("technical_node",    technical_node)
    g.add_node("oi_node",           oi_node)
    g.add_node("hints_node",        hints_node)
    g.add_node("correlation_node",  correlation_node)
    g.add_node("pattern_node",      pattern_node)
    g.add_node("prediction_node",   prediction_node)

    g.set_entry_point("orchestrator_node")

    # orchestrator → price (always first)
    g.add_edge("orchestrator_node", "price_node")

    # price → three parallel nodes
    g.add_edge("price_node", "technical_node")
    g.add_edge("price_node", "hints_node")
    g.add_edge("price_node", "oi_node")   # oi_node self-skips if not F&O

    # all three → correlation
    g.add_edge("technical_node", "correlation_node")
    g.add_edge("hints_node",     "correlation_node")
    g.add_edge("oi_node",        "correlation_node")

    # correlation → pattern → prediction → END
    g.add_edge("correlation_node", "pattern_node")
    g.add_edge("pattern_node",     "prediction_node")
    g.add_edge("prediction_node",  END)

    return g.compile()


graph = build_graph()
