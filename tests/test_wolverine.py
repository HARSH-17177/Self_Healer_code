import difflib
import json
import os
import shutil
import subprocess
import sys
import requests

from typing import List, Dict
from termcolor import cprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default model and retry config
DEFAULT_MODEL = os.environ.get("LLAMA_MODEL", "llama3.2:latest")
VALIDATE_JSON_RETRY = int(os.getenv("VALIDATE_JSON_RETRY", -1))
LLAMA_API_URL = "http://localhost:11434/api/chat"

# Read the system prompt
with open(os.path.join(os.path.dirname(__file__), "..", "prompt.txt"), "r") as f:
    SYSTEM_PROMPT = f.read()


def run_script(script_name: str, script_args: List) -> str:
    script_args = [str(arg) for arg in script_args]
    subprocess_args = (
        [sys.executable, script_name, *script_args]
        if script_name.endswith(".py")
        else ["node", script_name, *script_args]
    )

    try:
        result = subprocess.check_output(subprocess_args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as error:
        return error.output.decode("utf-8"), error.returncode
    return result.decode("utf-8"), 0


def llama_chat_completion(messages):
    response = requests.post(
        LLAMA_API_URL,
        json={"model": DEFAULT_MODEL, "messages": messages, "stream": False},
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def json_validated_response(
    model: str, messages: List[Dict], nb_retry: int = VALIDATE_JSON_RETRY
) -> Dict:
    json_response = {}
    if nb_retry != 0:
        try:
            content = llama_chat_completion(messages)
            messages.append({"role": "assistant", "content": content})
            json_start_index = content.index("[")
            json_data = content[json_start_index:]
            json_response = json.loads(json_data)
            return json_response

        except (json.decoder.JSONDecodeError, ValueError) as e:
            cprint(f"{e}. Re-running the query.", "red")
            cprint(f"\nLLAMA RESPONSE:\n\n{content}\n\n", "yellow")
            messages.append({
                "role": "user",
                "content": (
                    "Your response could not be parsed by json.loads. "
                    "Please restate your last message as pure JSON."
                ),
            })
            nb_retry -= 1
            return json_validated_response(model, messages, nb_retry)

        except Exception as e:
            cprint(f"Unknown error: {e}", "red")
            raise e

    raise Exception("No valid JSON response found after retries.")


def send_error_to_llama(
    file_path: str, args: List, error_message: str, model: str = DEFAULT_MODEL
) -> Dict:
    with open(file_path, "r") as f:
        file_lines = f.readlines()

    file_with_lines = []
    for i, line in enumerate(file_lines):
        file_with_lines.append(str(i + 1) + ": " + line)
    file_with_lines = "".join(file_with_lines)

    prompt = (
        "Here is the script that needs fixing:\n\n"
        f"{file_with_lines}\n\n"
        "Here are the arguments it was provided:\n\n"
        f"{args}\n\n"
        "Here is the error message:\n\n"
        f"{error_message}\n"
        "Please provide your suggested changes, and remember to stick to the "
        "exact format as described above."
    )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    return json_validated_response(model, messages)


def apply_changes(file_path: str, changes: List, confirm: bool = False):
    with open(file_path) as f:
        original_file_lines = f.readlines()

    operation_changes = [change for change in changes if "operation" in change]
    explanations = [change["explanation"] for change in changes if "explanation" in change]
    operation_changes.sort(key=lambda x: x["line"], reverse=True)

    file_lines = original_file_lines.copy()
    for change in operation_changes:
        operation = change["operation"]
        line = change["line"]
        content = change["content"]

        if operation == "Replace":
            file_lines[line - 1] = content + "\n"
        elif operation == "Delete":
            del file_lines[line - 1]
        elif operation == "InsertAfter":
            file_lines.insert(line, content + "\n")

    cprint("Explanations:", "blue")
    for explanation in explanations:
        cprint(f"- {explanation}", "blue")

    print("\nChanges to be made:")
    diff = difflib.unified_diff(original_file_lines, file_lines, lineterm="")
    for line in diff:
        if line.startswith("+"):
            cprint(line, "green", end="")
        elif line.startswith("-"):
            cprint(line, "red", end="")
        else:
            print(line, end="")

    if confirm:
        confirmation = input("Do you want to apply these changes? (y/n): ")
        if confirmation.lower() != "y":
            print("Changes not applied")
            sys.exit(0)

    with open(file_path, "w") as f:
        f.writelines(file_lines)
    print("Changes applied.")


def main(script_name, *script_args, revert=False, model=DEFAULT_MODEL, confirm=False):
    if revert:
        backup_file = script_name + ".bak"
        if os.path.exists(backup_file):
            shutil.copy(backup_file, script_name)
            print(f"Reverted changes to {script_name}")
            sys.exit(0)
        else:
            print(f"No backup file found for {script_name}")
            sys.exit(1)

    shutil.copy(script_name, script_name + ".bak")

    while True:
        output, returncode = run_script(script_name, script_args)

        if returncode == 0:
            cprint("Script ran successfully.", "blue")
            print("Output:", output)
            break
        else:
            cprint("Script crashed. Trying to fix...", "blue")
            print("Output:", output)
            json_response = send_error_to_llama(
                file_path=script_name,
                args=script_args,
                error_message=output,
                model=model,
            )

            apply_changes(script_name, json_response, confirm=confirm)
            cprint("Changes applied. Rerunning...", "blue")
