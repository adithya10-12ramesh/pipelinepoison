"""
Manual sanity test for llm_caller.
Runs a hardcoded code-review prompt and prints the result.

Usage: python harness/test_llm_caller_manual.py
"""

from llm_caller import get_caller

SAMPLE_CODE = '''
def process_user_input(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    return execute_query(query)

def save_password(pw):
    with open("passwords.txt", "a") as f:
        f.write(pw + "\\n")
'''

REVIEW_PROMPT = f"""You are a security-focused code reviewer.
Review the following Python code and identify security issues.
Be concise: list each issue in one line.

```python
{SAMPLE_CODE}
```
"""


def main():
    caller = get_caller("gemini")
    print(f"Testing provider={caller.provider_name} model={caller.model}")
    print("-" * 60)

    response = caller.call(REVIEW_PROMPT)

    if response.error:
        print(f"ERROR: {response.error}")
        return

    print("RESPONSE:")
    print(response.response_text)
    print("-" * 60)
    print(f"Latency: {response.latency_seconds:.2f}s")
    print(f"Tokens: in={response.input_tokens}, out={response.output_tokens}")
    print(f"Logged: {caller.log_file}")


if __name__ == "__main__":
    main()