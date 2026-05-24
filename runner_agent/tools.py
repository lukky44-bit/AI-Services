import json
import os
from typing import Optional

import requests
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq

class RunnerTools:
    """Encapsulates tool logic for generating and executing k6 scripts."""

    def __init__(self):
        model_name = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.llm = ChatGroq(model=model_name, temperature=temperature)

    @staticmethod
    def build_runner_agent_k6_prompt() -> str:
        prompt = """
You are a Senior Performance Engineer and k6 scripting expert.

Your task is to generate production-quality, executable k6 scripts.

--------------------------------------------------
SCRIPT GENERATION RULES
--------------------------------------------------

1. Understand the requested test type:

- smoke
- load
- stress
- spike
- soak
- endurance
- scenario-based

2. STRUCTURE SELECTION

A) Use:

export const options = {
  vus: X,
  duration: "Xs"
}

ONLY when ALL are true:
- simple smoke test
- fixed VUs
- single duration
- no ramping
- no hold phases
- no advanced behavior

B) Use SCENARIOS + EXECUTORS whenever ANY of these appear:

- ramp
- increase users
- decrease users
- hold users
- maintain load
- spike
- stages
- variable traffic
- load testing
- stress testing
- soak testing
- duration phases
- multiple traffic patterns

Preferred executor mapping:

Ramping traffic:
executor: "ramping-vus"

Fixed sustained traffic:
executor: "constant-vus"

Arrival-rate testing:
executor: "constant-arrival-rate"
executor: "ramping-arrival-rate"

IMPORTANT SCENARIO SYNTAX:
If you use an executor, you MUST put it inside the `scenarios` object. NEVER put `executor` directly in the root of `options`.
Example:
export const options = {
  scenarios: {
    my_scenario: {
      executor: "ramping-vus",
      startVUs: 0,
      stages: [
        { duration: "20s", target: 50 },
        { duration: "10s", target: 50 }
      ],
      gracefulRampDown: "2m"
    }
  },
  thresholds: {
    "http_req_duration": ["p(95)<2000"]
  }
};

--------------------------------------------------
CODE QUALITY RULES
--------------------------------------------------

Generate production-style k6 code:

Always use:

import http from "k6/http";
import { check, sleep } from "k6";

Use:

export const options = { ... }

NOT:
export let options

Include:
- thresholds (ONLY use valid metrics like `http_req_duration` or `http_req_failed`. NEVER use `http_status` in thresholds)
- checks (use `check()` for status codes like 200)
- realistic sleep
- comments for stages
- gracefulRampDown when ramping

Never generate:
- stagesInterval
- unnecessary vus field together with ramping-vus
- invalid k6 properties (like `executor` at the root of `options`)
- duplicate configuration

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY raw JavaScript.

No markdown.
No explanations.
No prose.

Generate directly executable k6 code.
"""
        return prompt.strip()

    def generate_k6_script(
        self,
        query: str,
    ) -> str:
        """
        Generates a valid k6 script based on the exact user query. 
        Returns the generated script.
        """
        prompt = self.build_runner_agent_k6_prompt()
        
        user_content = f"User Request:\n{query}"

        from langchain_core.messages import SystemMessage, HumanMessage
        response = self.llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=user_content),
        ])
        
        raw_output = response.content
        
        # Remove any markdown javascript or python code blocks to get the raw script
        script = raw_output.replace('```javascript', '').replace('```js', '').replace('```python', '').replace('```', '').strip()
        
        return script

    def execute_k6_script(self, script: str, vus: Optional[int] = None, run_id: Optional[str] = None) -> str:
        """
        Executes a k6 script by hitting a POST request to http://localhost:8081/run-test 
        with the script, vus, and an optional run_id in the body.
        """
        endpoint = os.getenv("RUNNER_TRIGGER_URL", "http://localhost:8081/run-test").strip()
        
        try:
            vus_val = int(vus) if vus else 1
        except Exception:
            vus_val = 1
            
        payload = {"vus": vus_val, "script": script}
        if run_id:
            payload["run_id"] = run_id

        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            try:
                raw = response.json()
            except Exception:
                raw = {"status_code": response.status_code, "text": response.text[:1000]}

            return json.dumps({
                "ok": response.status_code < 400,
                "status": str(response.status_code),
                "response": raw,
            })
        except Exception as exc:
            return json.dumps({"ok": False, "status": str(exc), "error": str(exc)})

    def get_tools(self) -> list[StructuredTool]:
        """Returns the list of tools configured for LangChain."""
        return [
            StructuredTool.from_function(
                func=self.generate_k6_script,
                name="generate_k6_script",
                description="Generates a valid k6 script based on the user's exact query. You must pass the ENTIRE user request string verbatim to the 'query' parameter."
            ),
            StructuredTool.from_function(
                func=self.execute_k6_script,
                name="execute_k6_script",
                description="Executes a generated k6 script by sending it to the runner service (http://localhost:8081/run-test). Accepts the script content, number of VUs, and an optional run_id. Returns the execution status and details."
            )
        ]
