# OKF assistant

Ask about the five papers in the OKF bundle (GPT-4 Technical Report, Sparks of AGI, HELM, the LLM survey, Foundation Models), find newer papers on arXiv, or turn the conversation into a report.

| ask | who answers |
|---|---|
| "What score did GPT-4 get on the bar exam?" | **OKF bundle** (okf_agent): answers from retrieved passages, with `[concept id]` citations |
| "Find recent papers on mixture of experts" | **arXiv scout** (paper_scout): up to 5 papers with links |
| "What does the bundle say about RLHF, and are there newer papers on it?" | both, side by side |
| "Write a report of our conversation" | **Report writer** (report_agent): a Markdown file you can download |

Under each answer, **Used okf_router** opens what happened: which agent the router picked and why, what each agent searched, what the model read, and what the **guardrails** blocked, masked or fixed.

The new-chat button (top left) starts a new conversation.
