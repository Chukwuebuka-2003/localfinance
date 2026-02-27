import os
import json
from typing import List, Dict, Any, Optional
from tools import (
    get_spending_by_category, get_largest_expenses, semantic_search_transactions,
    get_monthly_summary, compare_periods, get_income_by_source,
    detect_anomalies, get_transaction_frequency, get_category_trend,
    get_transactions_by_date_range, get_spending_velocity, get_running_balance,
    TOOL_REGISTRY
)

# Provider configuration
AI_PROVIDER = os.environ.get("AI_PROVIDER", "ollama").lower()  # "ollama" or "lmstudio"

# Ollama configuration
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite4:latest")

# LM Studio / OpenAI-compatible configuration
LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://host.docker.internal:1234")
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen/qwen3-vl-8b")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")

# Initialize clients based on provider
ollama_client = None
openai_client = None

if AI_PROVIDER == "ollama":
    try:
        import ollama
        ollama_client = ollama.Client(host=OLLAMA_HOST)
        print(f"Using Ollama provider at {OLLAMA_HOST} with model {OLLAMA_MODEL}")
    except Exception as e:
        print(f"Warning: Could not connect to Ollama: {e}")
elif AI_PROVIDER == "lmstudio":
    try:
        from openai import OpenAI
        openai_client = OpenAI(
            base_url=f"{LMSTUDIO_HOST}/v1",
            api_key=LMSTUDIO_API_KEY
        )
        print(f"Using LM Studio provider at {LMSTUDIO_HOST} with model {LMSTUDIO_MODEL}")
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI client for LM Studio: {e}")
else:
    print(f"Unknown AI_PROVIDER: {AI_PROVIDER}. Supported: 'ollama', 'lmstudio'")

# Tool definitions in OpenAI format (works for both Ollama and LM Studio)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_spending_by_category",
            "description": "Returns a breakdown of spending (negative transactions) grouped by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_largest_expenses",
            "description": "Returns the largest negative transactions (expenses).",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Limit of expenses to return."
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "semantic_search_transactions",
            "description": "Looks up specific transactions matching a query. Returns results broken down by month with monthly totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string (e.g. 'Amazon', 'restaurants', 'salary')."
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07'] for specific months. Results will be grouped by month."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_credit_debit",
            "description": "Calculates the exact total amount of money in (credit) and money out (debit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_by_description",
            "description": "Calculates the exact total amount of money sent to or received from a specific person, business, or entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The name of the entity, person, or business (e.g. 'John', 'Amazon')."
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipients",
            "description": "Returns a list of all names/entities the user has sent money to (debits/expenses). Use this when the user asks who they have sent money to or asks for a list of payees.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": []
            }
        }
    },
    # High Priority Tools
    {
        "type": "function",
        "function": {
            "name": "get_monthly_summary",
            "description": "Returns month-by-month breakdown of income, expenses, and net cash flow. Use this to show trends over multiple months or provide an overview of financial health by month.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": "Compare spending between two time periods (e.g., Jan vs Feb or Q1 vs Q2). Shows the difference in amount and percentage, with breakdown by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period1_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of date prefixes for first period (e.g., ['2025-01'])."
                    },
                    "period2_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of date prefixes for second period (e.g., ['2025-02'])."
                    }
                },
                "required": ["period1_prefixes", "period2_prefixes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_by_source",
            "description": "Groups income transactions by source/description. Use this to show where money is coming from (salary, transfers, refunds, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": []
            }
        }
    },
    # Medium Priority Tools
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Finds unusual transactions based on statistical outliers (amount/frequency). Flags transactions that are more than 2 standard deviations from the mean and detects potential duplicates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by, e.g., ['2025', '2026'] or ['2025-07']."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_frequency",
            "description": "Shows how often money is spent at a specific merchant or for a query. Calculates the average days between transactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for merchant or transaction type (e.g., 'Amazon', 'Netflix', 'groceries')."
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_category_trend",
            "description": "Track spending in one specific category over time. Shows month-by-month spending with trend indicator (increasing/decreasing/stable).",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Category name to track (e.g., 'Food', 'Transportation', 'Entertainment')."
                    },
                    "months": {
                        "type": "integer",
                        "description": "Number of months to look back (default: 6).",
                        "default": 6
                    }
                },
                "required": ["category"]
            }
        }
    },
    # Advanced Filtering Tools
    {
        "type": "function",
        "function": {
            "name": "get_transactions_by_date_range",
            "description": "Filter transactions by exact date range with optional semantic query. Use when user specifies exact start and end dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format."
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional search query to filter within results."
                    }
                },
                "required": ["start_date", "end_date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_spending_velocity",
            "description": "Calculate daily or weekly average spending rate. Shows how quickly money is being spent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    },
                    "period": {
                        "type": "string",
                        "enum": ["daily", "weekly"],
                        "description": "Analysis period - 'daily' or 'weekly'.",
                        "default": "daily"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_running_balance",
            "description": "Simulate account balance over time based on transaction amounts. Shows running total and identifies highest/lowest balance points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    },
    # Pattern Analysis Tools
    {
        "type": "function",
        "function": {
            "name": "get_day_of_week_analysis",
            "description": "Analyze spending patterns by day of week. Compares weekday (Mon-Fri) vs weekend (Sat-Sun) spending with detailed breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time_of_month_analysis",
            "description": "Analyze spending patterns by time of month. Compares early month (1-10), mid-month (11-20), and end-of-month (21-31) spending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_largest_expense_categories",
            "description": "Returns the largest expense categories with trend alerts. Shows top categories by total spending with month-over-month comparison.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top categories to show (default: 5).",
                        "default": 5
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_transactions",
            "description": "Find transactions similar to a query using semantic search. Groups related purchases and shows patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find similar transactions (e.g., 'Uber', 'groceries')."
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": ["query"]
            }
        }
    },
    # Merchant Analysis Tools
    {
        "type": "function",
        "function": {
            "name": "get_merchant_spending",
            "description": "Get total spending at a specific merchant over time. Shows transaction history and monthly totals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant": {
                        "type": "string",
                        "description": "Merchant name to search for (e.g., 'Amazon', 'Netflix')."
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": ["merchant"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_merchants",
            "description": "Returns the most frequented or highest-spend merchants. Ranks by total spending and transaction count.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of top merchants to show (default: 10).",
                        "default": 10
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_merchant_comparison",
            "description": "Compare spending between two merchants. Shows transaction counts, totals, and average transaction size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant1": {
                        "type": "string",
                        "description": "First merchant name to compare."
                    },
                    "merchant2": {
                        "type": "string",
                        "description": "Second merchant name to compare."
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": ["merchant1", "merchant2"]
            }
        }
    },
    # Recurring/Subscription Tools
    {
        "type": "function",
        "function": {
            "name": "detect_recurring_transactions",
            "description": "Identify subscriptions and recurring payments. Finds transactions with similar amounts on regular intervals.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_occurrences": {
                        "type": "integer",
                        "description": "Minimum number of occurrences to consider as recurring (default: 3).",
                        "default": 3
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_subscription_summary",
            "description": "List all recurring charges with monthly total. Focuses on monthly subscriptions and annual projections.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_payments",
            "description": "Predict upcoming recurring payments in the next N days. Based on detected recurring transaction patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to look ahead (default: 30).",
                        "default": 30
                    },
                    "date_prefixes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of date prefixes to filter by."
                    }
                },
                "required": []
            }
        }
    }
]


def _convert_messages_for_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert messages to OpenAI-compatible format."""
    converted = []
    for msg in messages:
        if msg.get("role") == "tool":
            # OpenAI uses 'tool' role for tool responses
            converted.append({
                "role": "tool",
                "content": msg.get("content", ""),
                "tool_call_id": msg.get("tool_call_id", msg.get("name", "unknown"))
            })
        elif "tool_calls" in msg:
            # Convert tool_calls format
            converted.append({
                "role": msg.get("role", "assistant"),
                "content": msg.get("content", ""),
                "tool_calls": msg["tool_calls"]
            })
        else:
            converted.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
    return converted


def _extract_tool_calls(response: Any) -> Optional[List[Dict]]:
    """Extract tool calls from response, handling both Ollama and OpenAI formats."""
    if AI_PROVIDER == "ollama":
        msg = response.get("message") if isinstance(response, dict) else response.message
        tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        return tool_calls
    else:  # lmstudio / openai
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            if hasattr(message, "tool_calls") and message.tool_calls:
                # Convert OpenAI tool calls to our internal format
                result = []
                for tc in message.tool_calls:
                    result.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        },
                        "type": "function"
                    })
                return result
        return None


def _extract_content(response: Any) -> Optional[str]:
    """Extract content from response, handling both Ollama and OpenAI formats."""
    if AI_PROVIDER == "ollama":
        msg = response.get("message") if isinstance(response, dict) else response.message
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        return content
    else:  # lmstudio / openai
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            return getattr(message, "content", None)
        return None


def _extract_role(response: Any) -> str:
    """Extract role from response."""
    if AI_PROVIDER == "ollama":
        msg = response.get("message") if isinstance(response, dict) else response.message
        return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "assistant")
    else:  # lmstudio / openai
        if hasattr(response, "choices") and response.choices:
            return "assistant"
        return "assistant"


def _make_chat_request(messages: List[Dict[str, Any]], use_tools: bool = True) -> Any:
    """Make a chat request to the configured provider."""
    if AI_PROVIDER == "ollama":
        if ollama_client is None:
            raise RuntimeError("Ollama client not initialized")
        return ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            tools=TOOLS if use_tools else None
        )
    else:  # lmstudio / openai
        if openai_client is None:
            raise RuntimeError("OpenAI client not initialized")

        openai_messages = _convert_messages_for_openai(messages)

        kwargs = {
            "model": LMSTUDIO_MODEL,
            "messages": openai_messages
        }
        if use_tools:
            kwargs["tools"] = TOOLS
            kwargs["tool_choice"] = "auto"

        return openai_client.chat.completions.create(**kwargs)


def _execute_tool(tool_call: Dict) -> tuple[str, str]:
    """Execute a tool call and return (result, tool_call_id)."""
    # Handle both Ollama and OpenAI formats
    func_data = tool_call.get("function") if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
    function_name = func_data.get("name") if isinstance(func_data, dict) else getattr(func_data, "name", None)
    arguments_str = func_data.get("arguments") if isinstance(func_data, dict) else getattr(func_data, "arguments", "{}")

    # Get tool call ID for OpenAI format
    tool_call_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", function_name)

    # Parse arguments
    try:
        if isinstance(arguments_str, str):
            kwargs = json.loads(arguments_str) if arguments_str else {}
        else:
            kwargs = arguments_str if arguments_str else {}
    except json.JSONDecodeError:
        kwargs = {}

    print(f"Executing tool {function_name} with args {kwargs}", flush=True)

    # Execute the tool
    if function_name in TOOL_REGISTRY:
        func_to_call = TOOL_REGISTRY[function_name]
        try:
            tool_result = func_to_call(**kwargs)
        except Exception as e:
            tool_result = f"Error executing tool {function_name}: {e}"
    else:
        tool_result = f"Error: Tool {function_name} not found."

    print(f"Tool {function_name} returned: {tool_result}", flush=True)
    return str(tool_result), tool_call_id


def chat_with_ai(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """
    Handles a conversation with the AI provider executing tools natively.
    Returns a tuple of (response_text, updated_messages_list)
    """

    # Ensure system prompt is set to guide behavior
    system_message = {
        "role": "system",
        "content": "You are a helpful personal finance AI assistant. You have access to tools that can query the user's uploaded bank statements and transactions. If a user asks a question about their spending, use the provided tools to fetch the data before answering. The currency is Naira (₦). Pay strict attention to the date requested by the user. If they ask for '2025 and 2026', pass date_prefixes as ['2025', '2026']. Do not assume or hallucinate a specific month (like '-07') unless the user explicitly asks for it. IMPORTANT: When searching for 'sender' or 'receiver', note that the transaction descriptions usually contain phrases like 'FROM [Name]' or 'TO [Name]'. Do not search for the literal word 'sender'; instead, search for specific names or use 'FROM' / 'TO' keywords in your query to find relevant entries."
    }

    if len(messages) == 0 or messages[0].get("role") != "system":
        messages.insert(0, system_message)

    # Check if provider is available
    if AI_PROVIDER == "ollama" and ollama_client is None:
        return "Error: Ollama client not initialized. Check OLLAMA_HOST configuration.", messages
    if AI_PROVIDER == "lmstudio" and openai_client is None:
        return "Error: LM Studio client not initialized. Check LMSTUDIO_HOST configuration.", messages

    try:
        # Loop to handle possible sequential tool call chains
        while True:
            # Step 1: Send messages + tool definitions to AI
            response = _make_chat_request(messages)

            # Step 2: Check if AI wants to call any tools
            tool_calls = _extract_tool_calls(response)
            content = _extract_content(response)
            role = _extract_role(response)

            if tool_calls:
                # Add the model's tool call request to the messages array
                # Format depends on provider
                if AI_PROVIDER == "ollama":
                    if isinstance(response.get("message") if isinstance(response, dict) else response.message, dict):
                        messages.append(response.get("message"))
                    else:
                        msg = response.message if hasattr(response, "message") else response.get("message")
                        messages.append({
                            "role": role,
                            "content": content or "",
                            "tool_calls": tool_calls
                        })
                else:  # lmstudio / openai
                    messages.append({
                        "role": "assistant",
                        "content": content or "",
                        "tool_calls": tool_calls
                    })

                # Execute all requested functions
                for tool_call in tool_calls:
                    tool_result, tool_call_id = _execute_tool(tool_call)

                    # Add the result to the messages array
                    # Format depends on provider
                    if AI_PROVIDER == "ollama":
                        func_data = tool_call.get("function") if isinstance(tool_call, dict) else getattr(tool_call, "function", None)
                        function_name = func_data.get("name") if isinstance(func_data, dict) else getattr(func_data, "name", "unknown")
                        messages.append({
                            "role": "tool",
                            "content": tool_result,
                            "name": function_name
                        })
                    else:  # lmstudio / openai
                        messages.append({
                            "role": "tool",
                            "content": tool_result,
                            "tool_call_id": tool_call_id
                        })

                # After appending tool results, let the while loop repeat to send them back to the LLM
                continue

            else:
                # Model didn't want to call tools, it's done and returned a response directly
                if not content or not str(content).strip():
                    content = "I've analyzed the data, but I couldn't formulate a textual response. Please check the logs."

                messages.append({"role": role, "content": content})

                print(f"Final AI Response: {content}", flush=True)
                return content, messages

    except Exception as e:
        return f"Error communicating with AI service: {str(e)}", messages


# Keep backward compatibility - alias the old function name
chat_with_granite = chat_with_ai
