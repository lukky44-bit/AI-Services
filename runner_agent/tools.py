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

Generate ONLY valid, executable, production-ready k6 JavaScript.

OUTPUT RULES

- Return ONLY raw JavaScript
- No markdown
- No explanations
- No prose
- Always output executable k6 JS
- Always include an export default function() { ... } block to execute the tests
- Always use:

import http from "k6/http";
import { check, sleep } from "k6";

- Always use:

export const options = { ... }

Never use:

export let options

CORE RULES

Preserve user requirements EXACTLY for traffic and load, BUT:
1. If the user provides a script containing invalid k6 properties (like `stagesDelay`, `summary`, etc.), you MUST DELETE them.
2. If the user provides a script missing an `export default function() { ... }` execution block, you MUST ADD one.
3. When using `scenarios` in the options block, you MUST DELETE any `vus`, `duration`, or `stages` properties located at the root of the options object. They conflict with scenarios.

Never modify valid traffic requirements:

- VUs
- durations
- thresholds
- endpoints
- stages
- executors
- arrival rates
- checks

User exclusions override defaults, EXCEPT for invalid properties or missing default functions.

Examples:

"don't use thresholds"
→ remove thresholds

"no checks"
→ remove check()

"no sleep"
→ remove sleep()

INTENT NORMALIZATION

Interpret ALL of these as ramping-vus:

volume test
check volume
volume capacity
volume growth
traffic growth
traffic scaling
load growth
capacity growth
step load
step-wise load
step-wise growth
step increase
progressive increase
gradually scale users
increment users
increase users
increase VUs
scale users
grow load
variable users
increase every X sec
increase VUs every X sec
scale users every X sec
ramp users

These ALWAYS mean:

executor:"ramping-vus"

Never use:

constant-vus

for these cases.

Interpret:

fixed users
constant users
steady load
maintain users
maintain traffic
soak test

As:

executor:"constant-vus"

Interpret:

requests/sec
rps
throughput
arrival rate

As:

executor:"constant-arrival-rate"

Interpret:

changing rps
ramping requests
variable requests

As:

executor:"ramping-arrival-rate"

Interpret:

exact iterations
fixed iterations

As:

executor:"shared-iterations"

MODE SELECTION

Use SIMPLE MODE ONLY:

export const options = {
   vus:X,
   duration:"Xs"
}

ONLY when ALL are true:

- smoke test
- fixed users
- single duration
- no ramping
- no hold phases
- no stages
- no advanced traffic

Otherwise ALWAYS use scenarios.

SCENARIO RULES

Executors MUST exist ONLY inside scenarios.

VALID:

export const options = {
   scenarios:{
      load_test:{
         executor:"ramping-vus"
      }
   }
}

TRANSFORMATION RULE:
If the user provides root `stages`, `vus`, or `duration`, you MUST move them INSIDE the scenarios block and DELETE them from the root.

Example Input:
export const options = { stages: [{ duration: "5s", target: 10 }] }

Example Output:
export const options = {
   scenarios: {
      load_test: {
         executor: "ramping-vus",
         startVUs: 0,
         stages: [{ duration: "5s", target: 10 }],
         gracefulRampDown: "5s"
      }
   }
}

FORBIDDEN:

root stages

export const options = {
   stages:[...]
}

root executor

export const options = {
   executor:"ramping-vus"
}

root duration with executor

root vus with executor

duplicate configs

duplicate stages

export const vus

export const iterations

EXECUTOR MAPPING

Ramping users:

executor:"ramping-vus"

Constant users:

executor:"constant-vus"

Fixed request rate:

executor:"constant-arrival-rate"

Variable request rate:

executor:"ramping-arrival-rate"

Fixed work:

executor:"shared-iterations"

EXECUTOR VALIDATION

constant-vus CANNOT contain:

stages
startVUs
target

If stages exist:

FORCE:

executor:"ramping-vus"

RAMP TEMPLATE

For ramping traffic ALWAYS generate:

export const options = {
   scenarios:{
      load_test:{
         executor:"ramping-vus",
         startVUs:0,
         stages:[
            ...
         ],
         gracefulRampDown:"5s"
      }
   }
}

RAMP INFERENCE

If user says:

increase every 5s

Infer staged growth.

Example:

increase VUs every 5s

Generate:

stages:[
 {duration:"5s",target:20},
 {duration:"5s",target:40},
 {duration:"5s",target:60},
 {duration:"5s",target:80},
 {duration:"5s",target:100}
]

ARRIVAL RATE RULES

For:

constant-arrival-rate
ramping-arrival-rate

Always include:

preAllocatedVUs

Optionally include:

maxVUs

Example:

executor:"constant-arrival-rate"
rate:100
timeUnit:"1s"
duration:"2m"
preAllocatedVUs:20
maxVUs:200

Never use:

timeUnit:"seconds"
gracefulRampDown (only valid for ramping-vus. Never include gracefulRampDown for arrival-rate or other executors!)

DEFAULT COMPONENTS

Default generate:

checks
sleep
thresholds

User exclusions override defaults.

CHECK RULES

Default:

check(response,{
   "status is 200":
   (r)=>r.status===200
})

Do not invent checks.

Forbidden:

body empty
status != 500
random validations

THRESHOLD RULES

Allowed:

http_req_duration
http_req_failed
checks
vus
iterations

Forbidden:

http_status

If user provides thresholds:

USE EXACT VALUES.

Never modify:

p95<1000

to

p95<1500

ANTI PATTERNS

Never generate:

root stages
root executor
stagesDelay
summary
stagesInterval
duplicate config
duplicate stages
unused imports
unused variables
unused code
extra functions
export function gracefulRampDown()
export const vus
export const iterations
gracefulRampDown (unless the executor is ramping-vus)

SELF VALIDATION

Before output verify:

1. Valid JavaScript
2. Runs in k6
3. Correct executor selected
4. No root stages
5. No root executor
6. User values preserved
7. User exclusions respected
8. No duplicate config
9. No invalid executor usage
10. Uses scenarios when not smoke

If invalid regenerate internally.
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
        
        # Robustly strip any conversational text before or after the code
        if 'import ' in script:
            script = script[script.find('import '):]
        if '}' in script:
            script = script[:script.rfind('}') + 1]
            
        # Programmatically remove gracefulRampDown if this is not a ramping-vus executor
        if "ramping-vus" not in script:
            cleaned_lines = []
            for line in script.splitlines():
                if "gracefulRampDown" not in line:
                    cleaned_lines.append(line)
            script = "\n".join(cleaned_lines)
            
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
                description="Generates a valid k6 script. You MUST pass the user's natural language instructions EXACTLY as they typed them into the 'query' parameter. DO NOT try to write or modify the script yourself here."
            )
        ]
