import difflib
import json
import os
import shutil
import subprocess
import sys
from typing import List, Dict
from termcolor import cprint

import ollama  # Make sure ollama is installed: pip install ollama

# Read the system prompt
with open(os.path.join(os.path.dirname(__file__), "..", "prompt.txt"), "r") as f:
    SYSTEM_PROMPT = f.read()

DEFAULT_MODEL = "llama3.2:latest"


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


def json_validated_response(model: str, messages: List[Dict], nb_retry: int = -1) -> Dict:
    json_response = {}
    while nb_retry != 0:
        full_prompt = "\n".join([m["content"] for m in messages])
        response = ollama.chat(model=model, messages=messages)
        content = response['message']['content']

        try:
            json_start_index = content.index("[")
            json_data = content[json_start_index:]
            json_response = json.loads(json_data)
            return json_response
        except Exception as e:
            cprint(f"Error: {e}. Re-asking...", "red")
            cprint(f"\nLLM Response:\n{content}\n", "yellow")
            messages.append({
                "role": "user",
                "content": "Your response could not be parsed by json.loads. Please restate it as valid pure JSON."
            })
            if nb_retry > 0:
                nb_retry -= 1

    raise Exception("No valid JSON response found.")


def send_error_to_gpt(file_path: str, args: List, error_message: str, model: str = DEFAULT_MODEL) -> Dict:
    with open(file_path, "r") as f:
        file_lines = f.readlines()

    file_with_lines = "".join(f"{i+1}: {line}" for i, line in enumerate(file_lines))

    prompt = (
        "Here is the script that needs fixing:\n\n"
        f"{file_with_lines}\n\n"
        "Here are the arguments it was provided:\n\n"
        f"{args}\n\n"
        "Here is the error message:\n\n"
        f"{error_message}\n"
        "Please provide your suggested changes in JSON format as described in the system prompt."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    return json_validated_response(model, messages)


def apply_changes(file_path: str, changes: List, confirm: bool = False):
    with open(file_path) as f:
        original_file_lines = f.readlines()

    operation_changes = [c for c in changes if "operation" in c]
    explanations = [c["explanation"] for c in changes if "explanation" in c]
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
        confirmation = input("Apply these changes? (y/n): ")
        if confirmation.lower() != "y":
            print("Changes not applied.")
            sys.exit(0)

    with open(file_path, "w") as f:
        f.writelines(file_lines)
    print("Changes applied.")


def main(*args, revert=False, model=DEFAULT_MODEL, confirm=False):
    if not args:
        print("Usage: python -m wolverine <script> <args>")
        sys.exit(1)

    script_name = args[0]
    script_args = args[1:]

    if revert:
        backup_file = script_name + ".bak"
        if os.path.exists(backup_file):
            shutil.copy(backup_file, script_name)
            print(f"Reverted {script_name}")
            sys.exit(0)
        else:
            print(f"No backup found for {script_name}")
            sys.exit(1)

    shutil.copy(script_name, script_name + ".bak")

    while True:
        output, returncode = run_script(script_name, script_args)

        if returncode == 0:
            cprint("Script ran successfully.", "green")
            print("Output:", output)
            break
        else:
            cprint("Script crashed. Attempting to fix...", "blue")
            print("Output:", output)
            changes = send_error_to_gpt(script_name, script_args, output, model=model)
            apply_changes(script_name, changes, confirm=confirm)
            cprint("Rerunning script...", "blue")
