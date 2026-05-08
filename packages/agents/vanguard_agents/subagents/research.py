"""Research Agent — uses local web_search and db_query tools.

These tools are read-only and have no side effects, so they live in-process
in ``vanguard_agents.tools`` rather than in the MCP server. The Research
Agent doesn't need to talk to MCP at all.
"""

from __future__ import annotations

import logging

from ..state import ConversationState
from ..tools import RESEARCH_TOOLS
from ._tool_loop import run_tool_loop, with_agent_badge

log = logging.getLogger(__name__)

RESEARCH_SYSTEM_PROMPT = """\
You are the Vanguard Research Agent. You answer questions about live market
data, fund performance, and the firm's internal data warehouse.

You have two tools (both run locally — no external services beyond a public
search API):

  - web_search(query, max_results): public web search for live data
                                    (rates, prices, news).
  - db_query(sql):                  read-only SELECT against the fund
                                    warehouse. Tables:
                                      funds(ticker, share_class, name,
                                            expense_ratio, asset_class,
                                            inception_year,
                                            net_assets_billions)
                                      fund_returns(ticker, year,
                                                   total_return)

When you need data, call the appropriate tool. Combine multiple tool calls
when useful (e.g. db_query first, then web_search to add context). Keep
your final answer concise and quote the values you retrieved. If a tool
returns an error, acknowledge it and offer an alternative.
"""


async def research_node(state: ConversationState) -> dict:
    final_text, new_messages, tool_results = await run_tool_loop(
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        tools=RESEARCH_TOOLS,
        user_message=state["user_message"],
        history=state.get("messages"),
    )
    decorated = with_agent_badge("Research Agent", final_text, tool_results)

    return {
        "final_response": decorated,
        "messages": new_messages,
        "tool_results": tool_results,
    }
